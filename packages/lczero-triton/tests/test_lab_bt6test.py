"""The lab reader on the BT6-test sponsor shape (CPU): `bt6_test_1024x28x16_triplet_cv4`.

1024 x 28, 16 heads x 64, sigmoid-GLU dff 1024 = d_model, `triplet_path` at six sites, and the four things no earlier
lab net carried: `ffn_softcap 12` (both GLU branches, blocks AND embedding), `rev_edge` (the site FFN's second
in-projection over the transposed rms), a 512-wide preprocess dense and a 512-wide policy head on the 1024 trunk.
Any export of that config serves -- random-init or a trained checkpoint; point `LC0EX_BT6TEST_NET` at one.
"""

import os
from pathlib import Path

import pytest
from lczero_triton.lab import _onnx
from lczero_triton.lab._mapping import ArchitectureError, _glu_by_dataflow, _softcap_behind, read_network
from lczero_triton.lab._names import plan_network

_NET = Path(os.environ.get(
    "LC0EX_BT6TEST_NET",
    str(Path.home() / "Kovax/nets/bt6_test_1024x28x16_triplet_cv4_randinit_vanilla_winner.pb.gz")))

pytestmark = pytest.mark.skipif(not _NET.exists(), reason=f"{_NET} not present")

_SITES = (3, 7, 11, 15, 19, 23)


@pytest.fixture(scope="module")
def loaded():
    _, graph = _onnx.load_carrier(_NET)
    return graph, read_network(graph)


def test_architecture_is_read_off_the_graph(loaded) -> None:
    _, network = loaded
    shape = network.architecture
    assert (shape.blocks, shape.d_model, shape.heads, shape.head_dim) == (28, 1024, 16, 64)
    assert (shape.ffn_hidden, shape.embedding_ffn_hidden) == (1024, 2048)
    assert shape.ffn_softcap == 12.0
    assert (shape.embedding_dense, shape.policy_width) == (512, 512)
    assert abs(network.policy_divisor - 512 ** 0.5) < 1e-4
    assert all(block.ffn_softcap == 12.0 for block in network.blocks) and network.embedding.ffn_softcap == 12.0


def test_every_site_carries_the_triplet_and_the_reverse_edge(loaded) -> None:
    graph, network = loaded
    egt = network.egt
    assert egt is not None and tuple(site.after_block for site in egt.sites) == _SITES
    for site in egt.sites:
        assert site.triplet is not None and site.triplet.contraction == "path"
        assert site.ffn_rev_weight is not None and site.ffn_rev_weight != site.ffn_in_weight
        assert tuple(graph.initializers[site.ffn_rev_weight].dims) == (1, egt.state_channels, egt.site_hidden)


def test_the_plan_names_the_new_tables_with_the_new_widths(loaded) -> None:
    _, network = loaded
    plans = {plan.name: plan for plan in plan_network(network)}
    for after in _SITES:
        assert plans[f"/encoder{after}/edge_site/ffn/dense1_rev/w"].shape == (64, 16)
    assert plans["/attn_body/preproc/w"].shape == (768, 64 * 512)
    assert plans["/attn_body/matmul/w"].shape == (112 + 512, 1024)
    assert plans["/policy/embedding/w"].shape == (1024, 512)
    assert plans["/policy/Q/w"].shape == plans["/policy/K/w"].shape == (512, 512)
    assert plans["/policy/promotion/w"].shape == (512, 4)


def test_an_unattributed_tanh_is_refused_not_dropped(loaded) -> None:
    """The guard behind `ffn_softcap`: the GLU walk reports the cap, and the cap pattern is what it says it is."""
    graph, network = loaded
    tangent = next(node for node in graph.nodes if node.op_type == "Tanh" and node.index > 100)
    closing = next(node for node in graph.nodes if node.op_type == "Mul" and tangent.outputs[0] in node.inputs)
    capped = _softcap_behind(graph, closing.outputs[0])
    assert capped is not None and capped[1] == 12.0
    start = tangent.index - 40
    glu = _glu_by_dataflow(graph, graph.nodes[start: tangent.index + 20])
    assert glu is not None and glu[3] == 12.0
    assert issubclass(ArchitectureError, ValueError)
