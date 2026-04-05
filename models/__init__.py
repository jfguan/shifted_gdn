from train.configs import ModelConfig, ModelType


def build_model(model_config: ModelConfig):
    if model_config.model == ModelType.GDN:
        from .gdn import GatedDeltaNet
        return GatedDeltaNet(model_config)
    elif model_config.model == ModelType.GDN_TOKENSHIFT:
        from .shifted_gdn import ShiftedGDN
        return ShiftedGDN(model_config)
    elif model_config.model == ModelType.TRANSFORMER:
        from .transformer import Transformer
        return Transformer(model_config)
    elif model_config.model == ModelType.TRANSFORMER_TS:
        from .shifted_transformer import ShiftedTransformer
        return ShiftedTransformer(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_config.model!r}")
