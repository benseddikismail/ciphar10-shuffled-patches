# Report

## Comparison of Test Accuracy and Loss (3 best models)

## Dataset creation and PCA analysis

## Transform and Plots

## Insights

## Experiments

There are many image classification models out there, but we experimented using the following models :

| Model Name | Test Accuracy | Test Loss | Test Accuracy 16x16| Test Loss 16x16|Test Accuracy 8x8| Test Loss 8x8|
|------------|---------------|-----------|---------------------|-----------------|------------------|-----------------|
| LeNet1    | 45.93         | 1.488     | 16.99 | 2.92 | 17.66 | 2.61 |
| LeNet2    | 54.7          | 1.324     | 17.32 | 2.249 | 14.68 | 2.278 |
| LeNet3    | 67.94         | 0.985   | 16.0 | 2.76 | 15.31 | 2.613 |
| LeNet3 (20 epochs)| 70.32 | 0.962 | 15.81 | 2.802 | 15.07 | 2.894 |
| Vgg      | 76.170.       | 1.643 |24.81 | 5.1   | 35.72  | 3.01 |
|ViT|62.960|1.126|35.58|2.1|31.67|2.2|

| Model Name | Architecture |
|------------|--------------|
| LeNet1    | LeNet1<br>(conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))<br>(conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))<br>(fc1): Linear(in_features=400, out_features=120, bias=True)<br>(fc2): Linear(in_features=120, out_features=84, bias=True)<br>(fc3): Linear(in_features=84, out_features=10, bias=True)<br>(flatten): Flatten(start_dim=1, end_dim=-1)<br>)<br>          |
| LeNet2    | LeNet2(<br>(conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1))<br>(bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>(conv2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))<br>(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,track_running_stats=True)<br>(fc1): Linear(in_features=1600, out_features=512, bias=True)<br>(fc2): Linear(in_features=512, out_features=256, bias=True)<br>(fc3): Linear(in_features=256, out_features=10, bias=True)<br>(flatten): Flatten(start_dim=1, end_dim=-1)<br>(dropout): Dropout(p=0.5, inplace=False)<br>) |
| LeNet3    | LeNet3(<br>(conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1))<br>(bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>(conv2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))<br>(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>(conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))<br>(bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>(avgpool): AdaptiveAvgPool2d(output_size=(1, 1))<br>(fc1): Linear(in_features=128, out_features=512, bias=True)<br>(fc2): Linear(in_features=512, out_features=256, bias=True)<br>(fc3): Linear(in_features=256, out_features=10, bias=True)<br>(flatten): Flatten(start_dim=1, end_dim=-1 ) <br>)|
| Vgg      | VGG(<br>(features): Sequential(<br>(0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(1): ReLU(inplace=True)<br>(2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(3): ReLU(inplace=True)<br>(4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)<br>(5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(6): ReLU(inplace=True)<br>(7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(8): ReLU(inplace=True)<br>(9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)<br>(10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(11): ReLU(inplace=True)<br>(12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(13): ReLU(inplace=True)<br>(14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(15): ReLU(inplace=True)<br>(16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)<br>(17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(18): ReLU(inplace=True)<br>(19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(20): ReLU(inplace=True)<br>(21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(22): ReLU(inplace=True)<br>(23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)<br>(24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(25): ReLU(inplace=True)<br>(26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(27): ReLU(inplace=True)<br>(28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(29): ReLU(inplace=True)<br>(30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)<br>)<br>(avgpool): AdaptiveAvgPool2d(output_size=(7, 7))<br>(classifier): Sequential(<br>(0): Linear(in_features=25088, out_features=4096, bias=True)<br>(1): ReLU(inplace=True)<br>(2): Dropout(p=0.5, inplace=False)<br>(3): Linear(in_features=4096, out_features=4096, bias=True)<br>(4): ReLU(inplace=True)<br>(5): Dropout(p=0.5, inplace=False)<br>(6): Linear(in_features=4096, out_features=10, bias=True)<br>)<br>)<br>. |
| VisionTransformer |<br>(patch_embedding): Conv2d(3, 128, kernel_size=(4, 4), stride=(4, 4))<br>(transformer_encoder): TransformerEncoder(num_layers=6, TransformerEncoderLayer(<br>&nbsp;&nbsp;&nbsp;&nbsp;(self_attn): MultiheadAttention(out_features=128, num_heads=8)<br>&nbsp;&nbsp;&nbsp;&nbsp;(linear1): Linear(in_features=128, out_features=512)<br>&nbsp;&nbsp;&nbsp;&nbsp;(linear2): Linear(in_features=512, out_features=128)<br>))<br>(fc): Linear(in_features=128, out_features=10)<br>|


### Description
The python script main.py trains a model on CIFAR10 dataset using PyTorch. It allows you to specify the number of epochs, model_class, batch size, learning rate, and L2 regularization strength.

### Example Command
```python
python main.py --epochs 10 --model_class 'Plain-Old-CIFAR10' --batch_size 128 --learning_rate 0.01 --l2_regularization 0.0001
```

### Options
- epochs (int): Number of epochs for training (default: 100).
- model_class (str): Model class name. Choices - 'Plain-Old-CIFAR10','D-shuffletruffle','N-shuffletruffle'. (default: 'Plain-Old-CIFAR10')
- batch_size (int): Batch size for training (default: 128).
- learning_rate (float): Learning rate for the optimizer (default: 0.01).
- l2_regularization (float): L2 regularization strength (default: 0.0).
