"""P0: config_builder must force 32GB-safe values, overriding dangerous lib defaults."""
import pytest

from engine.training.config_builder import build_t2v_config


@pytest.fixture()
def cfg(tmp_path):
    """Build a config with a real temporary model_path so ModelConfig.validate_model_path passes."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return build_t2v_config(
        model_path=str(model_dir),
        text_encoder_path="mlx-community/gemma-3-12b-it-4bit",
        preprocessed_data_root=str(tmp_path / "ds" / ".precomputed"),
        output_dir=str(tmp_path / "run"),
        steps=50,
        video_dims=(704, 480, 25),
    )


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
