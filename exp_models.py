import random

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # input channels: 3 (RGB), output channels: 6, kernel size: 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)  # input channels: 6, output channels: 16, kernel size: 5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # input features: 16*5*5, output features: 120
        self.fc2 = nn.Linear(120, 84) # input features: 120, output features: 84
        self.fc3 = nn.Linear(84, 10)  # input features: 84, output features: 10 (number of classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, (2, 2))  # Max pooling with kernel size 2x2
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, (2, 2))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet2(nn.Module):
    def __init__(self):
        super(LeNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)  # Increase output channels to 32
        self.bn1 = nn.BatchNorm2d(32)     # Add BatchNorm after conv1
        self.conv2 = nn.Conv2d(32, 64, 5) # Increase output channels to 64
        self.bn2 = nn.BatchNorm2d(64)     # Add BatchNorm after conv2
        self.fc1 = nn.Linear(64 * 5 * 5, 512)  # Increase output features of fc1
        self.fc2 = nn.Linear(512, 256)    # Increase output features of fc2
        self.fc3 = nn.Linear(256, 10)     # Output layer remains the same
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)    # Add dropout with a probability of 0.5

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))  # Apply BatchNorm after conv1
        x = torch.max_pool2d(x, (2, 2))
        x = torch.relu(self.bn2(self.conv2(x)))  # Apply BatchNorm after conv2
        x = torch.max_pool2d(x, (2, 2))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after fc1
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after fc2
        x = self.fc3(x)
        return x
    

class LeNet3(nn.Module):
    def __init__(self):
        super(LeNet3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)     # Increase output channels to 32
        self.bn1 = nn.BatchNorm2d(32)        # Add BatchNorm after conv1
        self.conv2 = nn.Conv2d(32, 64, 5)    # Increase output channels to 64
        self.bn2 = nn.BatchNorm2d(64)        # Add BatchNorm after conv2
        self.conv3 = nn.Conv2d(64, 128, 3)   # Additional convolutional layer
        self.bn3 = nn.BatchNorm2d(128)       # Add BatchNorm after conv3
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # AdaptiveAvgPool to ensure output size is (1, 1)
        self.fc1 = nn.Linear(128, 512)      # Adjusted input features of fc1
        self.fc2 = nn.Linear(512, 256)       # Adjusted output features of fc2
        self.fc3 = nn.Linear(256, 10)        # Output layer remains the same
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)       # Add dropout with a probability of 0.5

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))   # Apply BatchNorm after conv1
        x = torch.max_pool2d(x, (2, 2))
        x = torch.relu(self.bn2(self.conv2(x)))   # Apply BatchNorm after conv2
        x = torch.max_pool2d(x, (2, 2))
        x = torch.relu(self.bn3(self.conv3(x)))   # Apply BatchNorm after conv3
        x = self.avgpool(x)  # AdaptiveAvgPool to ensure output size is (1, 1)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after fc1
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after fc2
        x = self.fc3(x)
        return x
    
class LeNet4(nn.Module):
    def __init__(self):
        super(LeNet4, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)     # Increase output channels to 32
        self.bn1 = nn.BatchNorm2d(32)        # Add BatchNorm after conv1
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)    # Increase output channels to 64
        self.bn2 = nn.BatchNorm2d(64)        # Add BatchNorm after conv2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)   # Additional convolutional layer
        self.bn3 = nn.BatchNorm2d(128)       # Add BatchNorm after conv3
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)   # Additional convolutional layer
        self.bn4 = nn.BatchNorm2d(256)       # Add BatchNorm after conv4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # AdaptiveAvgPool to ensure output size is (1, 1)
        self.fc1 = nn.Linear(256, 512)      # Adjusted input features of fc1
        self.fc2 = nn.Linear(512, 256)       # Adjusted output features of fc2
        self.fc3 = nn.Linear(256, 128)       # Additional fully connected layer
        self.fc4 = nn.Linear(128, 10)        # Output layer remains the same
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)       # Add dropout with a probability of 0.5

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))   # Apply BatchNorm after conv1
        x = torch.max_pool2d(x, (2, 2))
        x = torch.relu(self.bn2(self.conv2(x)))   # Apply BatchNorm after conv2
        x = torch.max_pool2d(x, (2, 2))
        x = torch.relu(self.bn3(self.conv3(x)))   # Apply BatchNorm after conv3
        x = torch.max_pool2d(x, (2, 2))
        x = torch.relu(self.bn4(self.conv4(x)))   # Apply BatchNorm after conv4
        x = torch.max_pool2d(x, (2, 2))
        x = self.avgpool(x)  # AdaptiveAvgPool to ensure output size is (1, 1)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after fc1
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after fc2
        x = torch.relu(self.fc3(x))  # Additional fully connected layer
        x = self.dropout(x)  # Apply dropout after fc3
        x = self.fc4(x)
        return x
    
# Define the basic block for ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
class HybridViTCNN(nn.Module):
    def __init__(self, num_classes, image_size=224, patch_size=16, num_channels=3, num_layers=12, hidden_size=768, num_heads=12, mlp_dim=3072, dropout=0.1):
        super(HybridViTCNN, self).__init__()

        # Vision Transformer backbone
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)  # Pre-trained ViT backbone
        self.vit.head = nn.Identity()  # Replace the classification head with an identity layer
        
        # CNN layers for local feature extraction
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Final classification layers
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Forward pass through CNN layers
        cnn_features = self.cnn_layers(x)

        # Forward pass through ViT backbone
        vit_input = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        vit_features = self.vit(vit_input)

        # Ensure vit_features has the expected shape
        assert len(vit_features.shape) == 4, "Expected vit_features to be a 4D tensor"

        # Global average pooling
        vit_features = torch.mean(vit_features, dim=(2, 3))  # Calculate mean along dimensions 2 and 3

        # Final classification layers
        vit_features = self.dropout(vit_features)
        output = self.fc(vit_features)

        return output

    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Define max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 4x4 is the spatial dimension after 3 max pooling layers
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Define activation function
        self.relu = nn.ReLU()
        
        # Define dropout layer
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten the output from convolutional layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers with ReLU activation and dropout
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        
        # Output layer
        x = self.fc3(x)
        
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Assuming input image size is 32x32
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)  # Output size is 10 for classification

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten the output for fully connected layers
        x = x.view(-1, 32 * 8 * 8)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x


class DeiT_B(nn.Module):
    def __init__(self, num_classes=1000, image_size=224, patch_size=16, num_layers=12, embed_dim=768,
                 num_heads=12, mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.1, norm_layer=None):
        super(DeiT_B, self).__init__()

        # Patch embedding layer
        self.patch_embedding = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=drop_rate, activation='gelu'),
            num_layers=num_layers)

        # Classification head
        self.norm = norm_layer(embed_dim) if norm_layer else nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

        # Dropout
        self.dropout = nn.Dropout(p=drop_rate)

        # Initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embedding(x)  # [B, embed_dim, H', W']
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Positional embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_dim]
        x = x + self.pos_embedding[:, :(H * W + 1)]  # [B, num_patches + 1, embed_dim]

        # Transformer Encoder
        x = self.dropout(x)
        x = self.transformer_encoder(x)  # [B, num_patches + 1, embed_dim]

        # Classification head
        x = self.norm(x[:, 0])  # [B, embed_dim]
        x = self.fc(x)  # [B, num_classes]

        return x

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10)  # 10 classes for CIFAR10
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, num_classes=10, image_size=32, patch_size=4, num_channels=3, dim=128, depth=6, heads=8, mlp_dim=512):
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = num_channels * patch_size ** 2

        # Patch Embedding Layer
        self.patch_embedding = nn.Conv2d(num_channels, dim, kernel_size=patch_size, stride=patch_size)
        # Positional Embeddings
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Fully Connected Layer for Classification
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embedding(x)
        # Reshape to (batch_size, num_patches, dim)
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)  # Reshape and permute dimensions
        # Add Positional Embeddings
        x = x + self.positional_embedding[:, :x.size(1), :]
        # Transformer Encoder
        x = self.transformer_encoder(x)
        # Global Average Pooling
        x = x.mean(dim=1)
        # Classification
        x = self.fc(x)
        return x
