from reelrank.config import ROOT, Settings, load_settings


def test_settings_nested_access():
    cfg: Settings = load_settings()
    assert cfg.tmdb.fetch.max_movies > 0
    assert cfg.funnel.recall.merge_limit >= cfg.funnel.coarse.topk
    assert cfg.funnel.coarse.topk >= cfg.funnel.fine.topk
    assert len(cfg.ads.slots) > 0


def test_path_resolution_relative_to_root():
    cfg = load_settings()
    assert str(cfg.path("data/warehouse/probe.duckdb")).startswith(str(ROOT))
