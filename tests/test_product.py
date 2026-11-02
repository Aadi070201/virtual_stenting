"""The product's FFR pipeline settings (configs/product.yaml)."""

from vstent.inference import product
from vstent.physics.solver import PhysicsParams


def test_product_config_and_params(tmp_path):
    product.product_config.cache_clear()
    cfg = product.product_config(tmp_path / "missing.yaml")
    assert cfg.lesion_reader is None and cfg.outflow == "none" and cfg.wire
    path = tmp_path / "product.yaml"
    path.write_text("lesion_reader: null\noutflow: murray\nwire: false\n", encoding="utf-8")
    cfg = product.product_config(path)
    assert cfg.outflow == "murray" and not cfg.wire
    assert product.product_params(PhysicsParams(), cfg).outflow == "murray"
    assert product.product_reader(None) is None
    product.product_config.cache_clear()
    assert product.product_config().outflow in ("none", "murray")  # the committed file loads
