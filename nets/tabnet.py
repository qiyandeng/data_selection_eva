import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
import numpy as np
from nets.nets_utils import EmbeddingRecorder

class TabNet(nn.Module):
    def __init__(self, channel, num_classes, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False):
        if pretrained:
            raise NotImplementedError("Pretrained models not available.")
        super(TabNet, self).__init__()

        # Calculate the number of input features
        if isinstance(im_size, (tuple, list)):
            input_features = channel * np.prod(im_size)
        else:
            input_features = im_size * channel if channel != 1 else im_size
        
        # Define layers for the TabNet
        self.input_layer = nn.Linear(input_features, 128)
        self.shared_layer_1 = nn.Linear(128, 128)
        self.shared_layer_2 = nn.Linear(128, 128)
        
        self.attention_1 = nn.Linear(128, 1)
        self.attention_2 = nn.Linear(128, 1)
        
        self.decision_1 = nn.Linear(128, 128)
        self.decision_2 = nn.Linear(128, num_classes)
        
        # Final decision output
        self.fc = self.decision_2

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            # Flatten the input for a fully connected model
            out = torch.flatten(x, 1)  # Flatten the input
            
            # Shared layers (hidden layers)
            out = F.relu(self.input_layer(out))
            out = F.relu(self.shared_layer_1(out))
            out = F.relu(self.shared_layer_2(out))
            
            # Attention layers
            attention_1 = torch.sigmoid(self.attention_1(out))
            attention_2 = torch.sigmoid(self.attention_2(out))
            
            # Apply attention to the shared features
            out = out * attention_1 * attention_2
            
            # Decision layers
            out = F.relu(self.decision_1(out))
            out = self.fc(out)
        
        return out

    def feature_extractor(self, x):
        """Feature extractor: Returns the feature embedding after the shared layers and attention"""
        with set_grad_enabled(not self.no_grad):
            # Flatten the input
            out = torch.flatten(x, 1)
            
            # Shared layers (hidden layers)
            out = F.relu(self.input_layer(out))
            out = F.relu(self.shared_layer_1(out))
            out = F.relu(self.shared_layer_2(out))
            
            # Attention layers
            attention_1 = torch.sigmoid(self.attention_1(out))
            attention_2 = torch.sigmoid(self.attention_2(out))
            
            # Apply attention to the shared features
            out = out * attention_1 * attention_2
            
            # Return the feature embedding
            return self.embedding_recorder(out)
