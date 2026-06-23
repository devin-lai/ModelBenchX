from modelbenchx import cli


def test_csv_tuple_parses_trims_and_empties_to_none():
    assert cli._csv_tuple("resnet50, sam2 ,vit") == ("resnet50", "sam2", "vit")
    # Empty or whitespace-only input must collapse to None ("all"), not to an
    # empty tuple. An empty tuple would select nothing.
    assert cli._csv_tuple("") is None
    assert cli._csv_tuple("  ") is None
    assert cli._csv_tuple(" , , ") is None


def test_empty_filter_means_all_not_zero_runs():
    """`--backends ""` (or whitespace) must mean "all backends", not "none".
    Regression: an empty tuple overrode the None default and silently produced a
    zero-run benchmark."""
    for flag in ("--backends", "--modes", "--models"):
        args = cli.build_parser().parse_args(["run", flag, ""])
        cfg = cli._build_config(args)
        assert cfg.backends is None
        assert cfg.modes is None
        assert cfg.models is None
