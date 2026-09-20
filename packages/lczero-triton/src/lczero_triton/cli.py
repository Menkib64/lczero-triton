"""Build a BT4 computation graph from an Lc0 weights file."""

import argparse
import logging
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2

from lczero_triton.bt4._format import load_network
from lczero_triton.bt4.network import build

_LOGGER = logging.getLogger(__name__)
_AUTOTUNE_PRINT_ENV = "TRITON_PRINT_AUTOTUNING"
_DEFAULT_BATCH_SIZE = 169


def main(argv: Sequence[str] | None = None) -> int:
    """Build an Lc0 computation graph."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    _configure_logging()
    _LOGGER.info("loading network %s", arguments.network)
    network = load_network(arguments.network)
    builder = ExecutableBuilder(
        io_data_type=arguments.io_data_type or lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )
    _LOGGER.info("starting graph construction")
    with _autotune_progress(), redirect_stdout(sys.stderr):
        build(
            builder,
            network,
            batch_sizes=arguments.batch_sizes or [_DEFAULT_BATCH_SIZE],
        )
    _LOGGER.info("serializing executable to %s", arguments.output)
    builder.build_and_write(arguments.output)
    sys.stdout.write(f"{arguments.output}\n")
    return 0


def _configure_logging() -> None:
    """Configure human-readable application logs on stderr by default."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )


@contextmanager
def _autotune_progress() -> Iterator[None]:
    """Enable Triton's progress output without changing the caller's environment."""
    previous = os.environ.get(_AUTOTUNE_PRINT_ENV)
    if previous is None:
        os.environ[_AUTOTUNE_PRINT_ENV] = "1"
        _LOGGER.info("Triton autotuning progress is enabled")
    elif previous == "0":
        _LOGGER.info("Triton autotuning progress is disabled by environment")
    else:
        _LOGGER.info("Triton autotuning progress is enabled by environment")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_AUTOTUNE_PRINT_ENV, None)
        else:
            os.environ[_AUTOTUNE_PRINT_ENV] = previous


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    graph_parser = subparsers.add_parser(
        "graph",
        help="build an Lc0 computation graph from a weights file",
        allow_abbrev=False,
    )
    graph_parser.add_argument("--network", type=Path, required=True)
    graph_parser.add_argument("--output", type=Path, required=True)
    graph_parser.add_argument(
        "--batch-size",
        dest="batch_sizes",
        type=_parse_batch_size_expression,
        action="extend",
        default=None,
        help=(
            "fixed batch sizes to compile; use comma-separated values or "
            "inclusive ranges such as 1,2,4-16:2"
        ),
    )
    graph_parser.add_argument(
        "--io-data-type",
        dest="io_data_type",
        type=_parse_data_type,
        default="f16",
        help="data type for input/output buffers (f16 or f32, default: f16)"
    )
    return parser


def _parse_batch_size_expression(value: str) -> list[int]:
    """Parse comma-separated values and inclusive ranges."""
    sizes: list[int] = []
    for expression in value.split(","):
        bounds, _, step = expression.partition(":")
        start, *stops = map(int, bounds.split("-"))
        stop = stops[0] if stops else start
        sizes.extend(range(start, stop + 1, int(step or 1)))
    return sizes

def _parse_data_type(value: str) -> lc0ex_pb2.Buffer.DataType:
    """Parse a data type string into a Buffer.DataType enum."""
    value = value.lower()
    if value == "f16":
        return lc0ex_pb2.Buffer.DATA_TYPE_F16
    elif value == "f32":
        return lc0ex_pb2.Buffer.DATA_TYPE_F32
    else:
        raise argparse.ArgumentTypeError(f"invalid data type: {value}")

if __name__ == "__main__":
    raise SystemExit(main())
