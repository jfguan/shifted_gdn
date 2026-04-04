from train.configs import ModelConfig, ModelType


def build_model(model_config: ModelConfig):
    if model_config.model == ModelType.GDN:
        from .gated_deltanet import GatedDeltaNet
        return GatedDeltaNet(model_config)
    elif model_config.model == ModelType.GDN_TOKENSHIFT:
        from .shifted_gdn import GDNTokenShift
        return GDNTokenShift(model_config)
    elif model_config.model == ModelType.TRANSFORMER:
        from .transformer import Transformer
        return Transformer(model_config, token_shift=False)
    elif model_config.model == ModelType.TRANSFORMER_TS:
        from .transformer import Transformer
        return Transformer(model_config, token_shift=True)
    else:
        raise ValueError(f"Unknown model type: {model_config.model!r}")
