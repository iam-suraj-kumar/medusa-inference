from transformers import PretrainedConfig


class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model
    Args:
        medusa_num_layers: Number of Medusa layers 
        medusa_num_heads: Number of heads for each Medusa layer
        base_model_name_or_path: Name or path of the base model
        **kwargs: Additional keyword arguments to be passed to the parent class constructor
    """
    
    def __init__(
        self,
        medusa_num_layers=1,
        medusa_num_heads=5,
        base_model_name_or_path="./model/base-model/",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_layers = medusa_num_layers
        self.medusa_num_heads = medusa_num_heads
        self.base_model_name_or_path = base_model_name_or_path
