"""Build a 32GB-safe T2V LtxTrainerConfig (overrides dangerous library defaults)."""
from __future__ import annotations

from ltx_trainer_mlx.config import (
    CheckpointsConfig,
    DataConfig,
    FlowMatchingConfig,
    HubConfig,
    LoraConfig,
    LtxTrainerConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingStrategyConfig,
    ValidationConfig,
    WandbConfig,
)

# Locked to the set covered by the inference LoRA renaming map (§3.3 of the spec).
_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]


def build_t2v_config(
    *,
    model_path: str,
    text_encoder_path: str,
    preprocessed_data_root: str,
    output_dir: str,
    steps: int,
    rank: int = 32,
    learning_rate: float = 5e-4,
    seed: int = 42,
    video_dims: tuple[int, int, int] = (704, 480, 25),
) -> LtxTrainerConfig:
    """Construct a 32GB-safe T2V LtxTrainerConfig.

    Dangerous library defaults are overridden unconditionally:
    - batch_size forced to 1 (lib default 2 → OOM on 32GB)
    - enable_gradient_checkpointing forced True (lib default False)
    - validation.inference_steps forced to 8 (lib default 50 → very slow)
    - validation.generate_audio forced False (lib default True → extra RAM)
    - lora.target_modules locked to _TARGET_MODULES (matches inference rename map)

    Args:
        model_path: Path to the transformer weights directory.
        text_encoder_path: Path or HuggingFace ID for Gemma 3 12B text encoder.
        preprocessed_data_root: Root of the precomputed dataset (from ltx-prep).
        output_dir: Directory where checkpoints and logs are written.
        steps: Number of optimisation steps.
        rank: LoRA rank. Defaults to 32.
        learning_rate: AdamW learning rate. Defaults to 5e-4.
        seed: Global RNG seed. Defaults to 42.
        video_dims: Validation video dimensions (W, H, frames). Defaults to (704, 480, 25).

    Returns:
        A fully-constructed LtxTrainerConfig with 32GB-safe overrides applied.
    """
    return LtxTrainerConfig(
        model=ModelConfig(
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            training_mode="lora",
        ),
        lora=LoraConfig(
            rank=rank,
            alpha=rank,
            dropout=0.0,
            target_modules=list(_TARGET_MODULES),
        ),
        training_strategy=TrainingStrategyConfig(
            name="text_to_video",
            generate_audio=False,
        ),
        optimization=OptimizationConfig(
            learning_rate=learning_rate,
            steps=steps,
            batch_size=1,
            gradient_accumulation_steps=1,
            enable_gradient_checkpointing=True,
            optimizer_type="adamw",
            scheduler_params={},
        ),
        data=DataConfig(
            preprocessed_data_root=preprocessed_data_root,
        ),
        validation=ValidationConfig(
            prompts=["a serene landscape, cinematic"],
            video_dims=video_dims,
            inference_steps=8,
            generate_audio=False,
        ),
        checkpoints=CheckpointsConfig(),
        hub=HubConfig(),
        flow_matching=FlowMatchingConfig(
            timestep_sampling_params={},
        ),
        wandb=WandbConfig(
            tags=[],
        ),
        output_dir=output_dir,
        seed=seed,
    )
