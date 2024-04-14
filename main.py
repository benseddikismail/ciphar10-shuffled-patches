# Authors:
# (based on skeleton code for CSCI-B 657, Feb 2024)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from dataset_class import PatchShuffled_CIFAR10
from matplotlib import pyplot as plt
import argparse
import collections.abc as container_abcs
from itertools import repeat
from functools import partial

class CNN(nn.Module):

    def __init__(self):
        
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

# Define the model architecture for CIFAR10
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = CNN()
    def forward(self, x):
        return self.cnn(x) 

# class Net_D_shuffletruffle(nn.Module):
#     def __init__(self):
#         super(Net_D_shuffletruffle, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(64 * 8 * 8, 64)
#         self.fc2 = nn.Linear(64, 10)
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=64, num_heads=8)

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
        
#         # Self-attention
#         x = x.permute(0, 2, 3, 1)  # Change to (batch, height, width, channels)
#         x = x.view(-1, x.size(1) * x.size(2), x.size(3))  # Flatten the spatial dimensions
#         x = x.permute(1, 0, 2)  # Change to (seq_len, batch, embed_dim) for multihead attention
#         x, _ = self.multihead_attn(x, x, x)
#         x = x.permute(1, 0, 2)  # Change back to (batch, seq_len, embed_dim)
        
#         x = self.flatten(x)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# class Net_D_shuffletruffle(nn.Module):
#     def __init__(self):
#         super(Net_D_shuffletruffle, self).__init__()
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
#         # Max pooling layer
#         self.pool = nn.MaxPool2d(2, 2)
#         # Flatten layer
#         self.flatten = nn.Flatten()
#         # Fully connected layers
#         self.fc1 = nn.Linear(256 * 4 * 4, 512)  # Adjusted input size
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 10)
#         # Multi-head attention layer
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)

#     def forward(self, x):
#         # Convolutional layers with batch normalization and ReLU activation
#         x = self.pool(torch.relu(self.bn1(self.conv1(x))))
#         x = self.pool(torch.relu(self.bn2(self.conv2(x))))
#         x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        
#         # Self-attention
#         x = x.permute(0, 2, 3, 1)  # Change to (batch, height, width, channels)
#         x = x.view(-1, x.size(1) * x.size(2), x.size(3))  # Flatten the spatial dimensions
#         x = x.permute(1, 0, 2)  # Change to (seq_len, batch, embed_dim) for multihead attention
#         x, _ = self.multihead_attn(x, x, x)
#         x = x.permute(1, 0, 2)  # Change back to (batch, seq_len, embed_dim)
        
#         # Flatten and fully connected layers
#         x = self.flatten(x)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# helper functions
def trunc_normal_(tensor, mean=0., std=1.):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def _get_activation_fn(activation):
    
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class PatchInvarientPosEncoding(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, query, key, value):
        query = self.norm1(query)
        key = self.norm1(key)
        value = self.norm1(value)
        # import pdb
        # pdb.set_trace()
        src2 = self.self_attn(query, key, value)[0]

        src = query + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    pe = pe.flatten(1).transpose(0, 1)  # hw x d_model

    return pe.unsqueeze(0)  # 1 x hw x d_model


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B x HW x C
        # print(x.size())

        shuffle = []
        for idx in range(B):
            random_idx = torch.randperm(x.size(1))
            shuffle.append(x[idx][random_idx, :].unsqueeze(0))

        # print(random_idx)

        x = torch.cat(shuffle, dim=0)
        # print(x.size())

        return x


class HybridEmbed(nn.Module):
    """
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)


        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.reference = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.reference, std=.02)
        self.ref_encoding = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Sigmoid()
        )


        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
    
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        reference = self.reference.expand(B, -1, -1)


        ref = self.ref_encoding(x - reference)
        x = torch.cat((cls_tokens, x + ref), dim=1)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class Net_D_shuffletruffle(nn.Module):
    def __init__(self, *, img_size=32, patch_size=16, num_classes=10, embed_dim=384, depth=8, num_heads=8, mlp_ratio=3):
        super(Net_D_shuffletruffle, self).__init__()
        self.VisionTransformer = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )

    def forward(self, x):
        return self.VisionTransformer(x)

class Net_N_shuffletruffle(nn.Module):
    def __init__(self, *, img_size=32, patch_size=8, num_classes=10, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4):
        super(Net_N_shuffletruffle, self).__init__()
        self.VisionTransformer = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

    def forward(self, x):
        return self.VisionTransformer(x)

def eval_model(model, data_loader, criterion, device):
    # Evaluate the model on data from valloader
    correct = 0
    total = 0
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    
    return val_loss / len(data_loader), 100 * correct / len(data_loader.dataset)


def main(epochs = 100,
         model_class = 'Plain-Old-CIFAR10',
         batch_size = 128,
         learning_rate = 1e-4,
         l2_regularization = 0.0):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Load and preprocess the dataset, feel free to add other transformations that don't shuffle the patches.
    # (Note - augmentations are typically not performed on validation set)
    transform = transforms.Compose([
        transforms.ToTensor()])


    # Initialize training, validation and test dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000], generator=torch.Generator().manual_seed(0))

    # Normalize the data using mean and std of the training set
    train_mean = torch.stack([data[0] for data in trainset], dim=0).mean(axis=(0, 2, 3))
    train_std = torch.stack([data[0] for data in trainset], dim=0).std(axis=(0, 2, 3))

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=train_mean, std=train_std)
    ])


    # Apply transformation to trainset and valset
    # trainset.dataset.transform = transform
    # valset.dataset.transform = transform


    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Initialize Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size= batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Initialize the model, the loss function and optimizer
    if model_class == 'Plain-Old-CIFAR10':
        net = Net().to(device)
    elif model_class == 'D-shuffletruffle':
        net = Net_D_shuffletruffle().to(device)
    elif model_class == 'N-shuffletruffle':
        net = Net_N_shuffletruffle().to(device)

    print(net) # print model architecture
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = learning_rate, weight_decay= l2_regularization)

    train_accuracy_list = []
    val_accuracy_list = []
    train_loss_list = []
    val_loss_list = []

    # Train the model
    try:
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            net.train()
            for data in trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(trainloader)
            train_loss_list.append(train_loss)

            train_acc = 100 * correct / len(trainloader.dataset)
            train_accuracy_list.append(train_acc)

            val_loss, val_acc = eval_model(net, valloader, criterion, device)
            val_loss_list.append(val_loss)
            val_accuracy_list.append(val_acc)

            if epoch % 10 == 0:
                val_loss, val_acc = eval_model(net, valloader, criterion, device)
                print('epoch - %d loss: %.3f accuracy: %.3f val_loss: %.3f val_acc: %.3f' % (epoch, running_loss / len(trainloader), 100 * correct / len(trainloader.dataset), val_loss, val_acc))
            else:
                print('epoch - %d loss: %.3f accuracy: %.3f' % (epoch, running_loss / len(trainloader), 100 * correct / len(trainloader.dataset)))


        print('Finished training')
    except KeyboardInterrupt:
        pass

    # plt.figure(figsize=(10, 5))

    # plt.plot(range(epochs), train_loss_list, label='Training Loss')
    # plt.plot(range(epochs), val_loss_list, label='Validation Loss')  # Plot Validation Loss
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    # plt.show()


    # plt.figure(figsize=(10, 5))

    # plt.plot(range(epochs), train_accuracy_list, label='Training Accuracy')
    # plt.plot(range(epochs), val_accuracy_list, label='Validation Accuracy')  # Plot Validation Accuracy
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.legend()
    # plt.show()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    net.eval()
    # Evaluate the model on the test set
    test_loss, test_acc = eval_model(net, testloader, criterion, device)
    print('Test loss: %.3f accuracy: %.3f' % (test_loss, test_acc))

    # Evaluate the model on the patch shuffled test data

    patch_size = 16
    patch_shuffle_testset = PatchShuffled_CIFAR10(data_file_path = f'test_patch_{patch_size}.npz', transforms = transform)
    patch_shuffle_testloader = torch.utils.data.DataLoader(patch_shuffle_testset, batch_size=batch_size, shuffle=False)
    patch_shuffle_test_loss, patch_shuffle_test_acc = eval_model(net, patch_shuffle_testloader, criterion, device)
    print(f'Patch shuffle test loss for patch-size {patch_size}: {patch_shuffle_test_loss} accuracy: {patch_shuffle_test_acc}')

    patch_size = 8
    patch_shuffle_testset = PatchShuffled_CIFAR10(data_file_path = f'test_patch_{patch_size}.npz', transforms = transform)
    patch_shuffle_testloader = torch.utils.data.DataLoader(patch_shuffle_testset, batch_size=batch_size, shuffle=False)
    patch_shuffle_test_loss, patch_shuffle_test_acc = eval_model(net, patch_shuffle_testloader, criterion, device)
    print(f'Patch shuffle test loss for patch-size {patch_size}: {patch_shuffle_test_loss} accuracy: {patch_shuffle_test_acc}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', 
                        type=int, 
                        default= 100,
                        help= "number of epochs the model needs to be trained for")
    parser.add_argument('--model_class', 
                        type=str, 
                        default= 'Plain-Old-CIFAR10', 
                        choices=['Plain-Old-CIFAR10','D-shuffletruffle','N-shuffletruffle'],
                        help="specifies the model class that needs to be used for training, validation and testing.") 
    parser.add_argument('--batch_size', 
                        type=int, 
                        default= 100,
                        help = "batch size for training")
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default = 0.001,
                        help = "learning rate for training")
    parser.add_argument('--l2_regularization', 
                        type=float, 
                        default= 0.0,
                        help = "l2 regularization for training")
    
    args = parser.parse_args()
    main(**vars(args))
