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

| Model Name | Architecture |
|------------|--------------|
| LeNet1    | LeNet1<br>(conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))<br>(conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))<br>(fc1): Linear(in_features=400, out_features=120, bias=True)<br>(fc2): Linear(in_features=120, out_features=84, bias=True)<br>(fc3): Linear(in_features=84, out_features=10, bias=True)<br>(flatten): Flatten(start_dim=1, end_dim=-1)<br>)<br>          |
| LeNet2    | LeNet2(<br>(conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1))<br>(bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>(conv2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))<br>(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,track_running_stats=True)<br>(fc1): Linear(in_features=1600, out_features=512, bias=True)<br>(fc2): Linear(in_features=512, out_features=256, bias=True)<br>(fc3): Linear(in_features=256, out_features=10, bias=True)<br>(flatten): Flatten(start_dim=1, end_dim=-1)<br>(dropout): Dropout(p=0.5, inplace=False)<br>) |
| LeNet3    | LeNet3(<br>(conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1))<br>(bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>(conv2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))<br>(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>(conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))<br>(bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>(avgpool): AdaptiveAvgPool2d(output_size=(1, 1))<br>(fc1): Linear(in_features=128, out_features=512, bias=True)<br>(fc2): Linear(in_features=512, out_features=256, bias=True)<br>(fc3): Linear(in_features=256, out_features=10, bias=True)<br>(flatten): Flatten(start_dim=1, end_dim=-1 ) <br>)|



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
