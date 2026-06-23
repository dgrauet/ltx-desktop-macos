"""P0: the ltx-trainer dependency must be importable in the backend env."""


def test_ltx_trainer_imports():
    import ltx_trainer_mlx  # noqa: F401


def test_trainer_symbols_exist():
    import ltx_trainer_mlx.trainer as t

    # LtxvTrainer lives in the .trainer submodule (not exported from the package root).
    assert hasattr(t, "LtxvTrainer"), (
        "expected LtxvTrainer in ltx_trainer_mlx.trainer"
    )
