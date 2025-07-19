import torch
from torch import nn

class EmbeddingExtractor(nn.Module):
    """
    A generic feature extractor that wraps a neural network and returns
    the output of all layers except the final classification head.

    Example usage:
        model = nets.__dict__[model_name](...)
        extractor = EmbeddingExtractor(model)
        embeddings = extractor(input_tensor)
    """
    def __init__(self, model: nn.Module):
        super(EmbeddingExtractor, self).__init__()
        # Extract all layers except the final "classifier" or "fc" layer
        # We assume the final layer is the last child module
        children = list(model.children())
        if len(children) <= 1:
            raise ValueError("Model has no feature layers to extract.")
        # Remove last layer
        self.features = nn.Sequential(*children[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass input through all feature layers
        x = self.features(x)
        # Flatten if needed (e.g., after global pooling layers)
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        return x
