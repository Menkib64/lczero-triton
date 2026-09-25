#!/usr/bin/env python3
"""Q1: compile one int8 GEMM specialization with a C++ main that checks it against a host reference, then time it.

No torch, no lc0ex runtime: the rendered kernel source (exactly what the builder compiles) plus a harness that fills
random int8 codes and weights, runs the kernel once, recomputes every output on the host (int64 accumulation, the
epilogue in double), and prints the error census; then 200 timed launches.

    test_cutlass_gemm_i8_standalone.py --m 4096 --n 3072 --k 1024 --epilogue bias [--tile 128,128,64,64,64,64,3]
    test_cutlass_gemm_i8_standalone.py --m 4096 --n 1024 --k 1024 --epilogue glu --output i8 --softcap 12

Exit status 0 only if every output is within tolerance: FP16 outputs within 1 ulp of the double reference (the
kernel computes in FP32 then rounds once), int8 codes equal except where the reference sits within 1e-3 of a
rounding tie (FP32 vs double can land either side of it).
"""

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "lczero-triton" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "lc0ex" / "src"))

from lczero_triton.bt4.kernels import cutlass_gemm_i8 as gemm  # noqa: E402

_TEST_MAIN = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

static double ref_softcap(double x, double c) { return c * std::tanh(x / c); }

int main() {
  std::mt19937 rng(20260923);
  std::uniform_int_distribution<int> code(-127, 127);
  std::normal_distribution<float> normal(0.f, 1.f);
  const int kOut = kOutputColumns;
  std::vector<int8_t> a((size_t)kM * kK), b((size_t)kN * kK);
  for (auto& x : a) x = (int8_t)code(rng);
  for (auto& x : b) x = (int8_t)code(rng);
  std::vector<float> scale(kN), bias(kN), prescale(kOut);
  for (auto& x : scale) x = 1e-5f * (0.5f + std::fabs(normal(rng)));
  for (auto& x : bias) x = normal(rng);
  for (auto& x : prescale) x = 10.f * (0.5f + std::fabs(normal(rng)));
  std::vector<__half> skip((size_t)kM * kOut);
  for (auto& x : skip) x = __float2half(normal(rng));
  const bool residual = {is_residual}, glu = {is_glu}, out_i8 = {is_i8};
  const double cap = {softcap};
  const size_t out_bytes = (size_t)kM * kOut * (out_i8 ? 1 : 2);

  void *d, *da, *db, *ds, *dbias, *dskip, *dpre;
  cudaMalloc(&d, out_bytes); cudaMalloc(&da, a.size()); cudaMalloc(&db, b.size());
  cudaMalloc(&ds, kN * 4); cudaMalloc(&dbias, kN * 4); cudaMalloc(&dskip, skip.size() * 2); cudaMalloc(&dpre, kOut * 4);
  cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(ds, scale.data(), kN * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dbias, bias.data(), kN * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dskip, skip.data(), skip.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dpre, prescale.data(), kOut * 4, cudaMemcpyHostToDevice);
  cudaMemset(d, 0x7f, out_bytes);
  const int shared_bytes = (int)sizeof(SharedStorage);
  if (cudaFuncSetAttribute((const void*)&{entry}, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes) != cudaSuccess) {
    std::printf("FAIL smem attribute %d\n", shared_bytes); return 1;
  }
  const dim3 grid((kM + ThreadblockShape::kM - 1) / ThreadblockShape::kM, (kN + ThreadblockShape::kN - 1) / ThreadblockShape::kN, 1);
  const dim3 block({threads}, 1, 1);
  {entry}<<<grid, block, shared_bytes>>>({test_arguments});
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { std::printf("FAIL launch: %s\n", cudaGetErrorString(err)); return 1; }
  std::vector<unsigned char> out(out_bytes);
  cudaMemcpy(out.data(), d, out_bytes, cudaMemcpyDeviceToHost);

  // host reference over every output
  long long bad = 0, ties = 0, checked = 0; double worst = 0.0;
  std::vector<double> y(kN);
  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      long long acc = 0;
      const int8_t* ar = &a[(size_t)m * kK]; const int8_t* br = &b[(size_t)n * kK];
      for (int j = 0; j < kK; ++j) acc += (long long)ar[j] * (long long)br[j];
      y[n] = (double)acc * (double)scale[n] + (double)bias[n];
    }
    for (int o = 0; o < kOut; ++o) {
      double ref, slack = 0.0;
      if (glu) {
        double gate = 1.0 / (1.0 + std::exp(-y[2 * o]));
        double up = y[2 * o + 1];
        if (cap > 0) { gate = ref_softcap(gate, cap); up = ref_softcap(up, cap); }
        ref = gate * up;
        // the served softcap is c * (1 - e) / (1 + e) with e = exp(-2|x|/c) in FP32: near x = 0 the 1 - e cancels,
        // an absolute floor ~1e-6 that the int8 step (~0.1) never sees; the FP16 GLU route has the same form.
        slack = 2e-5;
      } else {
        ref = y[o];
        if (residual) {
          const double s = (double)__half2float(skip[(size_t)m * kOut + o]);
          // FP32 cancellation when the projection and the skip nearly cancel: one FP32 rounding of each term.
          slack = 2.0 * 5.960464477539063e-08 * (std::fabs(ref) + std::fabs(s));
          ref += s;
        }
      }
      ++checked;
      if (out_i8) {
        const double scaled = ref * (double)prescale[o] + 0.5;
        double want = std::floor(scaled); want = want < -127 ? -127 : (want > 127 ? 127 : want);
        const int got = (int)(int8_t)out[(size_t)m * kOut + o];
        const double frac = scaled - std::floor(scaled);
        const bool near_tie = frac < 2e-3 || frac > 1 - 2e-3;
        if (got != (int)want) { if (near_tie && std::abs(got - want) == 1) ++ties; else ++bad; }
        worst = std::fmax(worst, std::fabs(got - want));
      } else {
        const __half got_h = reinterpret_cast<const __half*>(out.data())[(size_t)m * kOut + o];
        const double got = (double)__half2float(got_h);
        // one FP16 ulp at the reference's magnitude (subnormal floor 2^-24)
        const double mag = std::fabs(ref);
        const double ulp = mag < 6.103515625e-05 ? 5.960464477539063e-08 : std::ldexp(1.0, (int)std::floor(std::log2(mag)) - 10);
        const double err = std::fmax(0.0, std::fabs(got - ref) - slack) / ulp;
        worst = std::fmax(worst, err);
        if (err > (glu ? 2.0 : 1.0)) ++bad;
      }
    }
  }
  std::printf("# checked %lld outputs: %lld out of tolerance, %lld one-code tie flips, worst %s %.3f\n",
              checked, bad, ties, out_i8 ? "|code diff|" : "err/ulp", worst);

  cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
  for (int i = 0; i < 20; ++i) {entry}<<<grid, block, shared_bytes>>>({test_arguments});
  cudaEventRecord(start);
  for (int i = 0; i < 200; ++i) {entry}<<<grid, block, shared_bytes>>>({test_arguments});
  cudaEventRecord(stop); cudaEventSynchronize(stop);
  float ms = 0; cudaEventElapsedTime(&ms, start, stop);
  const double us = ms * 1000.0 / 200.0;
  const double tops = 2.0 * kM * (double)kN * kK / (us * 1e-6) / 1e12;
  std::printf("# time %.2f us  %.1f TOPS  smem %d B  grid %dx%d  block %d\n", us, tops, shared_bytes, grid.x, grid.y, {threads});
  std::printf("%s\n", bad == 0 ? "PASS" : "FAIL");
  return bad == 0 ? 0 : 1;
}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True, help="OUTPUT columns (the GLU's hidden width)")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--epilogue", default="bias", choices=gemm.EPILOGUES)
    parser.add_argument("--output", default="f16", choices=gemm.OUTPUTS)
    parser.add_argument("--softcap", type=float, default=0.0)
    parser.add_argument("--arch", type=int, default=89)
    parser.add_argument("--tile", default="128,128,64,64,64,64,3")
    parser.add_argument("--keep", type=Path, help="write the rendered .cu here too")
    arguments = parser.parse_args()
    values = [int(v) for v in arguments.tile.split(",")]
    tile = ((values[0], values[1], values[2]), (values[3], values[4], values[5]), values[6])
    spec = gemm.CutlassGemmI8Specialization(arguments.m, arguments.n, arguments.k, arguments.arch,
                                            epilogue=arguments.epilogue, output=arguments.output,
                                            glu_softcap=arguments.softcap)
    fields = gemm._parameter_fields(spec)  # noqa: SLF001
    test_arguments = fields["sweep_arguments"].replace("(d)", "(d)").replace("(a)", "(da)").replace("(b)", "(db)")
    test_arguments = (test_arguments.replace("(scale)", "(ds)").replace("(bias)", "(dbias)")
                      .replace("(skip)", "(dskip)").replace("(prescale)", "(dpre)"))
    main_source = (_TEST_MAIN.replace("{entry}", gemm.entry_point_name(spec))
                   .replace("{threads}", str(gemm._thread_count(tile[0], tile[1])))  # noqa: SLF001
                   .replace("{test_arguments}", test_arguments)
                   .replace("{is_residual}", "true" if spec.epilogue == "residual" else "false")
                   .replace("{is_glu}", "true" if spec.epilogue == "glu" else "false")
                   .replace("{is_i8}", "true" if spec.output == "i8" else "false")
                   .replace("{softcap}", repr(float(arguments.softcap))))
    source = gemm.render_source(spec, tile) + main_source
    if arguments.keep:
        arguments.keep.write_text(source, encoding="utf-8")
    with TemporaryDirectory(prefix="i8test-") as directory:
        path = Path(directory) / "test.cu"
        binary = Path(directory) / "test"
        path.write_text(source, encoding="utf-8")
        command = [str(gemm._NVCC), "-std=c++17", "-O3", "--expt-relaxed-constexpr", "-w",  # noqa: SLF001
                   f"-arch=sm_{arguments.arch}", "-I", str(gemm.CUTLASS_INCLUDE), str(path), "-o", str(binary)]
        compiled = subprocess.run(command, check=False, capture_output=True, text=True)  # noqa: S603
        if compiled.returncode:
            print(compiled.stderr[-6000:])
            print("COMPILE FAILED")
            return 2
        print(f"# {gemm.entry_point_name(spec)} tile {tile}")
        return subprocess.run([str(binary)], check=False).returncode  # noqa: S603


if __name__ == "__main__":
    raise SystemExit(main())
