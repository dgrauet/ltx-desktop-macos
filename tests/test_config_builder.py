"""P0 + P1: config_builder must force 32GB-safe values and support low_ram flag."""
import pytest

from engine.training.config_builder import build_t2v_config


@pytest.fixture()
def cfg(tmp_path):
    """Build a low_ram=True config so P0 tests retain their original semantics."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return build_t2v_config(
        model_path=str(model_dir),
        text_encoder_path="mlx-community/gemma-3-12b-it-4bit",
        preprocessed_data_root=str(tmp_path / "ds" / ".precomputed"),
        output_dir=str(tmp_path / "run"),
        steps=50,
        video_dims=(704, 480, 25),
        low_ram=True,
    )


# ---------------------------------------------------------------------------
# P0 tests (low_ram=True path — original assertions preserved)
# ---------------------------------------------------------------------------

def test_forces_batch_size_1(cfg):
    assert cfg.optimization.batch_size == 1


def test_forces_gradient_checkpointing_on(cfg):
    assert cfg.optimization.enable_gradient_checkpointing is True


def test_forces_low_validation_and_no_audio(cfg):
    assert cfg.validation.video_dims == (704, 480, 25)
    assert cfg.validation.inference_steps <= 8
    assert cfg.validation.generate_audio is False


def test_locks_target_modules(cfg):
    assert cfg.lora.target_modules == ["to_k", "to_q", "to_v", "to_out.0"]


# ---------------------------------------------------------------------------
# P1 tests: low_ram flag semantics
# ---------------------------------------------------------------------------

def _cfg(low_ram: bool, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    return build_t2v_config(
        model_path=str(model_dir),
        text_encoder_path="mlx-community/gemma-3-12b-it-4bit",
        preprocessed_data_root=str(tmp_path / "pc"),
        output_dir=str(tmp_path / "out"),
        steps=10,
        video_dims=(704, 480, 25),
        low_ram=low_ram,
    )


def test_low_ram_forces_safe_values(tmp_path):
    c = _cfg(True, tmp_path)
    assert c.optimization.batch_size == 1
    assert c.optimization.enable_gradient_checkpointing is True


def test_normal_mode_is_default(tmp_path):
    c = _cfg(False, tmp_path)
    assert c.optimization.batch_size == 2
    assert c.optimization.enable_gradient_checkpointing is False


def test_target_modules_locked_in_both_modes(tmp_path):
    for lr in (True, False):
        assert _cfg(lr, tmp_path).lora.target_modules == ["to_k", "to_q", "to_v", "to_out.0"]
