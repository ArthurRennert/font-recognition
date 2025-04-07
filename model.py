"""
Model Module

This module defines an enhanced ResNet18-based model for font classification.
It uses a pretrained ResNet18 backbone to extract features, removes its original final
fully connected layer, and adds new fully connected layers with Batch Normalization and
Dropout for improved regularization.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Enhanced(nn.Module):
    def __init__(self, num_classes=7, dropout_prob=0.5, pretrained=True):
        """
        Initializes the enhanced ResNet18 model.

        Args:
            num_classes (int): Number of output classes.
            dropout_prob (float): Dropout probability.
            pretrained (bool): If True, load pretrained weights from ImageNet.
        """
        super(ResNet18Enhanced, self).__init__()
        # Load ResNet18 with pretrained weights if requested.
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # Remove the original final fully connected layer.
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Add new fully connected layers with BatchNorm and Dropout.
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Output logits for each class.
        """
        # Extract features using the pretrained ResNet18 backbone.
        features = self.resnet(x)  # Expected shape: [B, in_features]
        x = self.dropout(features)
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_resnet18_model(num_classes=7, dropout_prob=0.5, pretrained=True):
    """
    Returns an instance of the enhanced ResNet18 model.

    Args:
        num_classes (int): Number of output classes.
        dropout_prob (float): Dropout probability.
        pretrained (bool): If True, load pretrained weights from ImageNet.

    Returns:
        nn.Module: The enhanced ResNet18 model.
    """
    return ResNet18Enhanced(num_classes=num_classes, dropout_prob=dropout_prob, pretrained=pretrained)
