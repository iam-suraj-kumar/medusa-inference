import logging

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from trl import SFTTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class MedusaSFTTrainer(SFTTrainer):
    def __init__(
        self,
        medusa_num_heads,
        medusa_heads_coefficient,
        medusa_decay_coefficient,
        medusa_scheduler,
        train_only_medusa_heads,
        medusa_lr_multiplier,
        **kwargs,
    ):
        self.medusa_num_heads = medusa_num_heads
        self.medusa_heads_coefficient = medusa_heads_coefficient
        self.medusa_decay_coefficient = medusa_decay_coefficient
        self.medusa_scheduler = medusa_scheduler
        self.train_only_medusa_heads = train_only_medusa_heads
        self.medusa_lr_multiplier = medusa_lr_multiplier

        if getattr(kwargs["model"], "is_quantized", False) and train_only_medusa_heads:
            # Trainer does not know that we will freeze base model layers, and would
            # raise an error that we need an adapter when model is quantized
            # So we trick it to think it is not quantized during init
            setattr(kwargs["model"], "is_quantized", False)
            super().__init__(**kwargs)
            setattr(kwargs["model"], "is_quantized", True)
        else:
            super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        """
        Compute the training loss for the model. Overriding the default with custom loss for Medusa
        Args:
            model: The model for which to compute the loss.
            inputs: The input data, including input IDs, attention mask, and labels.
            return_outputs: Whether to return model outputs along with the loss.
        Returns:
            The computed loss, optionally with model outputs.
        """

        logits = model(
            **inputs,
            train_only_medusa_heads=self.train_only_medusa_heads,
        )
        labels = inputs["labels"]
        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
        num_heads = logits.shape[0]
        for i in range(num_heads):
            medusa_logits = logits[i, :, : -(1 + i)].contiguous()
            medusa_labels = labels[..., 1 + i:].contiguous()
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])
            medusa_labels = medusa_labels.view(-1)
            medusa_labels = medusa_labels.to(medusa_logits.device)
            loss_i = loss_fct(medusa_logits, medusa_labels)
            # Compute the coefficient for medusa losses
            if self.medusa_scheduler == "constant":
                medusa_scheduler_coefficient = 1
            else:
                raise ValueError(
                    f"Invalid medusa_scheduler: {self.medusa_scheduler}. "
                    "Must be 'constant'."
                )
            if i == 0:
                if not self.train_only_medusa_heads:
                    loss += loss_i
            else:
                loss += (
                    loss_i
                    * self.medusa_decay_coefficient**i
                    * self.medusa_heads_coefficient
                    * medusa_scheduler_coefficient
                )
            not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
            medusa_labels = medusa_labels[not_ignore]

            # Add top-k accuracy
            for k in range(1, 10):
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                log[f"medusa{i}_top{k}"] = correct.float().mean().item()

            log[f"medusa{i}_loss"] = loss_i.item()
            log["medusa_scheduler_coefficient"] = medusa_scheduler_coefficient
            logger.debug(log)
        return (loss, logits) if return_outputs else loss

    def create_optimizer(self):
        """
        Overriding default method, the only change is in optimizer_grouped_parameters
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            # Separately set lr for medusa_head (this is the only change here)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n in decay_parameters
                            and p.requires_grad
                            and "medusa_head" not in n
                        )
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n in decay_parameters
                            and p.requires_grad
                            and "medusa_head" in n
                        )
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * self.medusa_lr_multiplier,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        return self.optimizer
