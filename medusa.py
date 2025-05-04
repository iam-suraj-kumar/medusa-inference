import logging
import types
from contextlib import nullcontext
from typing import List, Optional

import torch

from .layers import medusa_layer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def add_medusa_heads(
    model,
    medusa_num_layers,
    medusa_num_heads,
):
    """
    Args:
        model: Base language model to be used.
        medusa_num_heads: Number of additional tokens to predict
    """
    hidden_size = model.lm_head.weight.shape[-1]
    vocab_size = model.lm_head.weight.shape[0]

    model.config.medusa_num_layers = medusa_num_layers
    model.config.medusa_num_heads = medusa_num_heads

    model.medusa_num_heads = medusa_num_heads
    model.medusa_heads = medusa_layer(medusa_num_heads, hidden_size, vocab_size)

    # Ensure medusa_head's dtype and device align with the base_model
    model.medusa_heads.to(model.dtype).to(model.device)
    logger.info(f"Loading medusa heads in {str(model.dtype)} to device {model.device}")

    for i in range(medusa_num_heads):
        # Initialize the weights of each medusa_head using the base model's head weights
        model.medusa_heads[i][-1].weight.data[:] = model.lm_head.weight.data[:]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train_only_medusa_heads: bool = False,
    ):
        """Forward pass of the MedusaModel
        Returns:
            A tensor containing predictions from all Medusa heads
        """
        maybe_grad = torch.no_grad() if train_only_medusa_heads else nullcontext()
        with maybe_grad:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            # lm head output
            medusa_logits = [self.lm_head(hidden_states)]
        for i in range(self.medusa_num_heads):
            # medusa head  output
            medusa_logits.append(self.medusa_heads[i](hidden_states))
        return torch.stack(medusa_logits, dim=0)

    model.forward = types.MethodType(forward, model)
    