"""P0: pin the real ltx-trainer API so config_builder binds to truth, not guesses.

Introspected from ltx_trainer_mlx on 2026-06-23. If this test fails after an
upstream ltx-2-mlx bump, the config_builder overrides in engine/training/ must
be revisited before shipping.
"""
import inspect

import pytest

from ltx_trainer_mlx.trainer import LtxvTrainer


def _config_cls():
    from ltx_trainer_mlx.config import LtxTrainerConfig
    assert LtxTrainerConfig is not None, "config class not found"
    return LtxTrainerConfig


def test_config_class_import_path():
    """Confirm exact import path: ltx_trainer_mlx.config.LtxTrainerConfig (not Ltxv...)."""
    from ltx_trainer_mlx.config import LtxTrainerConfig
    assert LtxTrainerConfig.__module__ == "ltx_trainer_mlx.config"
    assert LtxTrainerConfig.__name__ == "LtxTrainerConfig"


def test_config_has_expected_top_level_fields():
    fields = _config_cls().model_fields
    for name in ("model", "lora", "optimization", "validation", "output_dir", "seed",
                 "training_strategy", "data", "checkpoints", "hub", "flow_matching", "wandb"):
        assert name in fields, f"expected top-level config field {name!r}; got {list(fields)}"


def test_top_level_defaults():
    fields = _config_cls().model_fields
    assert fields["seed"].default == 42
    assert fields["output_dir"].default == "outputs"
    # lora is optional (default None)
    assert fields["lora"].default is None


def test_optimization_dangerous_defaults():
    """If these change upstream, config_builder overrides must be revisited."""
    from ltx_trainer_mlx.config import OptimizationConfig
    opt = OptimizationConfig.model_fields
    assert opt["batch_size"].default == 2
    assert opt["enable_gradient_checkpointing"].default is False
    assert opt["learning_rate"].default == 0.0005
    assert opt["steps"].default == 3000
    assert opt["gradient_accumulation_steps"].default == 1


def test_validation_defaults():
    """video_dims and inference_steps defaults -- callers rely on these."""
    from ltx_trainer_mlx.config import ValidationConfig
    val = ValidationConfig.model_fields
    assert val["video_dims"].default == (960, 544, 97)
    assert val["inference_steps"].default == 50
    assert val["frame_rate"].default == 25.0
    assert val["guidance_scale"].default == 4.0
    assert val["seed"].default == 42
    assert val["interval"].default == 100
    assert val["skip_initial_validation"].default is False


def test_lora_config_defaults():
    from ltx_trainer_mlx.config import LoraConfig
    lc = LoraConfig.model_fields
    assert lc["rank"].default == 64
    assert lc["alpha"].default == 64
    assert lc["dropout"].default == 0.0
    assert lc["target_modules"].default == ["to_k", "to_q", "to_v", "to_out.0"]


def test_checkpoints_defaults():
    from ltx_trainer_mlx.config import CheckpointsConfig
    ck = CheckpointsConfig.model_fields
    assert ck["interval"].default is None
    assert ck["keep_last_n"].default == 1
    assert ck["precision"].default == "bfloat16"


def test_trainer_train_signature():
    """train() must accept step_callback — Task 6 (train_runner) depends on it."""
    sig = inspect.signature(LtxvTrainer.train)
    params = sig.parameters
    assert "step_callback" in params, (
        f"train() has no step_callback param; got params: {list(params)}"
    )
    # step_callback is optional (default None)
    assert params["step_callback"].default is None
    # disable_progress_bars is the other param
    assert "disable_progress_bars" in params


def test_trainer_train_return_annotation():
    """train() returns (Path, TrainingStats) — callers unpack this."""
    sig = inspect.signature(LtxvTrainer.train)
    ann = sig.return_annotation
    assert ann != inspect.Parameter.empty, "train() must have a return annotation"
    # The annotation is a string 'tuple[Path, TrainingStats]' in this version
    assert "TrainingStats" in str(ann)
    assert "Path" in str(ann)


def test_step_callback_signature():
    """StepCallback is Callable[[int, int, list[Path]], None] — callers must match this."""
    from ltx_trainer_mlx.trainer import StepCallback
    # StepCallback is a type alias via collections.abc.Callable
    # Real shape: Callable[[int, int, list[Path]], None] -- pin the full 3-arg arity
    # so a drift to e.g. (int, Path) fails loudly (Task 6 train_runner depends on it).
    cb_str = str(StepCallback)
    assert "list[pathlib.Path]" in cb_str, (
        f"expected 'list[pathlib.Path]' (3rd callback arg) in StepCallback type, got: {cb_str}"
    )
    assert cb_str.count("int") >= 2, (
        f"expected two 'int' args (current_step, total_steps) in StepCallback type, "
        f"got: {cb_str}"
    )


def test_training_stats_fields():
    """TrainingStats fields — callers read these after train() returns."""
    from ltx_trainer_mlx.trainer import TrainingStats
    ann = getattr(TrainingStats, "__annotations__", {})
    for field in ("total_time_seconds", "steps_per_second", "samples_per_second",
                  "peak_memory_gb", "batch_size"):
        assert field in ann, f"TrainingStats missing field {field!r}; got {list(ann)}"


def test_lora_checkpoint_filename_pattern():
    """LoRA output: <output_dir>/checkpoints/lora_weights_step_NNNNN.safetensors.

    The filename is built as f"{prefix}_weights_step_{step:05d}.safetensors" where
    prefix = "lora" for LoRA mode. This is a static check against trainer.py source.
    """
    src = inspect.getsource(LtxvTrainer._save_checkpoint)
    # Verify the dynamic prefix assignment: prefix = "lora" if is_lora else "model"
    assert '"lora"' in src, (
        "expected 'lora' prefix string in _save_checkpoint source"
    )
    # Verify the f-string template for the filename
    assert "_weights_step_" in src, (
        "expected '_weights_step_' in filename template in _save_checkpoint source"
    )
    assert ".safetensors" in src, (
        "expected .safetensors extension in _save_checkpoint source"
    )
    assert "checkpoints" in src, (
        "expected 'checkpoints' subdirectory in _save_checkpoint source"
    )
    # Verify 5-digit zero-padded step number format
    assert ":05d}" in src, (
        "expected :05d zero-padding in step filename; got different padding"
    )
