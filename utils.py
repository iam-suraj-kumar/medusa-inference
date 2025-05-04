import logging
from dataclasses import dataclass, field

from safetensors.torch import save_file

from .config import MedusaConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def freeze_layers(model):
    """
    Freeze base model layers
    """
    logger.info("Freeze layers")
    for param in model.parameters():
        param.requires_grad = False

    for param in model.medusa_heads.parameters():
        param.requires_grad = True
    logger.info("Finished freezing layers")


def save_medusa_heads(final_save_dir, medusa_num_heads, model, torch_dtype):
    """
    Save medusa config, medusa heads
    """
    # Save medusa config
    medusa_config = MedusaConfig(
        medusa_num_heads=medusa_num_heads,
        base_model_name_or_path="./model/base-model/",
        version="2",
    )
    medusa_config.save_pretrained(final_save_dir)

    # Save medusa heads
    model.medusa_heads.to(torch_dtype)
    logger.info(f"Converting medusa heads to {str(torch_dtype)}")

    state_dict = model.medusa_heads.state_dict()
    save_file(
        state_dict,
        f"{final_save_dir}/medusa_lm_head.safetensors",
    )


@dataclass
class ScriptArguments:
    train_dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the train dataset"
        },
    )
    eval_dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the eval dataset"
        },
    )
    model_path: str = field(
        default=None, metadata={"help": "Path to the fine-tuned model"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )
    medusa_num_heads: int = field(
        default=5,
        metadata={"help": "Number of heads for the Medusa"},
    )
    medusa_heads_coefficient: float = field(
        default=0.1, metadata={"help": "Coefficient for Medusa heads"}
    )
    medusa_decay_coefficient: float = field(
        default=1.0, metadata={"help": "Decay coefficient for Medusa"}
    )
    medusa_scheduler: str = field(
        default="constant",
        metadata={
            "help": "Scheduler type for Medusa, currently only constant is supported"
        },
    )
    train_only_medusa_heads: bool = field(
        default=True, metadata={"help": "If True, train only medusa heads"}
    )
    medusa_lr_multiplier: float = field(
        default=1.0, metadata={"help": "Learning rate multiplier for Medusa"}
    )


