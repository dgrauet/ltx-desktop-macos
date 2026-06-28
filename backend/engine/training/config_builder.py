"""Build a T2V LtxTrainerConfig with optional 32GB-safe overrides."""
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
    low_ram: bool = False,
    enable_validation: bool = False,
) -> LtxTrainerConfig:
    """Construct a T2V LtxTrainerConfig with optional 32GB-safe overrides.

    Always overridden (both modes):
    - validation.inference_steps = 8 (lib default 50 → very slow)
    - validation.generate_audio = False (lib default True → extra RAM)
    - lora.target_modules locked to _TARGET_MODULES (matches inference rename map)

    low_ram=True (opt-in, safe for 32GB / quantized models):
    - batch_size = 1
    - enable_gradient_checkpointing = True

    low_ram=False (default, normal training):
    - batch_size = 2
    - enable_gradient_checkpointing = False

    enable_validation=False (default): validation is fully disabled
    (interval=0, no prompts). The initial step-0 validation is a ~2-min
    sustained-Metal inference that reliably trips the macOS GPU watchdog
    ("Impacting Interactivity" SIGKILL) and keeps the VAE decoder resident,
    raising peak memory for the first training step. Samples are a UI
    placeholder we do not consume, so off-by-default is the safe choice on
    32GB. Set enable_validation=True to generate periodic preview samples.

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
        low_ram: If True, force batch_size=1 and gradient checkpointing for 32GB safety.
            Defaults to False.

    Returns:
        A fully-constructed LtxTrainerConfig.
    """
    batch_size = 1 if low_ram else 2
    grad_ckpt = bool(low_ram)

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
            batch_size=batch_size,
            gradient_accumulation_steps=1,
            enable_gradient_checkpointing=grad_ckpt,
            optimizer_type="adamw",
            scheduler_params={},
        ),
        data=DataConfig(
            preprocessed_data_root=preprocessed_data_root,
        ),
        validation=ValidationConfig(
            # Disable validation fully when enable_validation is False:
            #  - empty prompts -> has_validation False, so the trainer never loads
            #    the text encoder, feature extractor, or VAE decoder (it gates them
            #    on `interval and prompts`), lowering peak memory;
            #  - skip_initial_validation removes the step-0 inference (the
            #    sustained-Metal burst that trips the macOS GPU watchdog);
            #  - a huge interval guarantees the periodic block (gated on interval
            #    only) never fires. interval must be > 0 (pydantic constraint), so
            #    0 is not an option.
            prompts=["a serene landscape, cinematic"] if enable_validation else [],
            interval=100 if enable_validation else 1_000_000_000,
            skip_initial_validation=not enable_validation,
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
