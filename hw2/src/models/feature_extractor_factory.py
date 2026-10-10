from .feature_extractor import FEATURE_EXTRACTOR_MODES, FeatureExtractor
from .feature_extractor_linear_conv import FeatureExtractorLinearConv
from .feature_extractor_lower_linear_conv import FeatureExtractorLowerLinearConv


def factory_feature_extractor(
    fe_type: FEATURE_EXTRACTOR_MODES = "linearconv",
) -> type[FeatureExtractor]:
    """Класс экстрактора по имени из конфига."""
    if fe_type == "linearconv":
        return FeatureExtractorLinearConv
    if fe_type == "lowrlinearconv":
        return FeatureExtractorLowerLinearConv
    raise NotImplementedError
