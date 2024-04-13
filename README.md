# Report

## Comparison of Test Accuracy and Loss (3 best models)

## Dataset creation and PCA analysis

We wrote a script to set up a dataset for our image classification task using CIFAR-10. We randomly selected 25 images from the CIFAR-10 test dataset. These images will be the core of our dataset.Then, we decided to add some variety to our dataset using data augmentation. We created shuffled versions of each of those 25 images. We divided each image into patches of two different sizes, 16x16 and 8x8. After that, we shuffled these patches randomly and put them back together to form a new image. 
Finally,We combined them with the original images to make our final dataset. This mix of original and shuffled images adds more diversity to our dataset..


## Transform and Plots

## Insights

## Experiments

### Approach

<details>
<summary>LeNet1 Architecture Overview (click to expand)</summary> 
  
  ### Thought process behind choosing this model:
   - **Historical Significance**: Pioneering CNN for basic image classification tasks like MNIST.
   - **Simplicity and Efficiency**: Simple architecture suitable for low-resolution images and limited resources.
  
  1. **Convolutional Layers (conv1 and conv2)**:
     - **Purpose**: Extract hierarchical features like edges and textures from input images.
     - **Effect**: Learnable filters convolve with input images, capturing local patterns and enhancing feature representation.

  2. **Pooling Layers (not explicitly mentioned)**:
     - **Purpose**: Reduce spatial dimensions and extract dominant features.
     - **Effect**: Downsample feature maps, preserving essential information while reducing computational complexity and preventing overfitting.

  3. **Fully Connected Layers (fc1, fc2, and fc3)**:
     - **Purpose**: Perform classification based on learned features.
     - **Effect**: Neurons in these layers learn to associate features with specific classes, facilitating accurate classification.

  4. **Flatten Layer**:
     - **Purpose**: Reshape feature maps for input to fully connected layers.
     - **Effect**: Converts multi-dimensional feature maps into a flat vector, maintaining spatial relationships and enabling further processing.

  5. **Overall Effect**:
     - LeNet1 sequentially extracts features, reduces dimensionality, and performs classification.
     - By leveraging convolutional, pooling, fully connected layers, and flattening, it effectively learns hierarchical representations and makes accurate predictions on image data.
</details>

<details>
  <summary>LeNet2 Architecture Overview (click to expand)</summary>
  
  ### Thought process behind choosing this model:
   - **Enhanced Performance**: Adds more layers and batch normalization for improved accuracy.
   - **Balanced Complexity**: Strikes a balance between complexity and efficiency.
  
  1. **Convolutional Layers**:
      - LeNet2 has increased the number of output channels in its convolutional layers compared to LeNet1, allowing for more diverse feature extraction.

2. **Batch Normalization**:
    - LeNet2 incorporates batch normalization layers after each convolutional layer to stabilize and accelerate training.

3. **Fully Connected Layers**:
    - LeNet2 adjusts the input size to its first fully connected layer based on the output size of preceding layers, potentially enhancing feature processing capabilities.

4. **Dropout Regularization**:
    - LeNet2 introduces a dropout layer with a dropout rate of 0.5 after the fully connected layers to prevent overfitting and improve model generalization.

5. **Overall Effect**:
    - LeNet2 builds upon LeNet1 with additional layers and adjustments, aiming to improve feature extraction, training stability, and generalization capabilities. These changes are expected to result in a more powerful and robust model for image classification tasks.
  
  ![LeNet2](https://media.github.iu.edu/user/24623/files/a67f2f83-edb8-4689-a4ad-adc2b49dd694)


</details>

<details>
  <summary>LeNet3 Architecture Overview (click to expand)</summary>
  
### Thought process behind choosing this model:
   - **Architectural Refinement**: Introduces adaptive pooling for diverse image characteristics.
   - **Hierarchical Feature Extraction**: Designed to capture hierarchical features effectively.
  
1. **Convolutional Layers**:
   - LeNet3 introduces an additional convolutional layer compared to LeNet2, resulting in a deeper network architecture.

2. **Batch Normalization**:
   - Similar to LeNet2, LeNet3 incorporates batch normalization layers after each convolutional layer to stabilize and accelerate training.

3. **Pooling Layer**:
   - LeNet3 utilizes adaptive average pooling instead of traditional pooling layers, enabling the network to handle input images of variable sizes and aspect ratios.

4. **Fully Connected Layers**:
   - LeNet3 maintains a similar configuration of fully connected layers as LeNet2, with adjustments in input and output sizes to accommodate the changes in the convolutional layers.

5. **Overall Effect**:
   - LeNet3 further extends the architecture of LeNet2 by adding more convolutional layers and adaptive pooling, potentially improving its ability to capture hierarchical features and generalize to diverse image datasets.
  
  ### 10 epochs
  ![LeNet3_10epoch](https://media.github.iu.edu/user/24623/files/65b8ac15-ce83-43d2-a6f0-f745b4f02551)
  
  ### 30 epochs
  ![LeNet3_epoch30_plain](https://media.github.iu.edu/user/24623/files/af34e906-e900-4a9f-9b7c-e530d9945d65)

  </details>

<details>
  <summary>LeNet4 Architecture Overview (click to expand)</summary>
  
### Thought process behind choosing this model:
   - **Increased Capacity**: Adds more layers to handle highly complex datasets.
   - **Balancing Complexity**: Explores advanced architectures while maintaining efficiency.
  
1. **Additional Convolutional Layers**:
   - Additional convolutional layers (conv3 and conv4) with increased output channels capture more complex features.
   
2. **Batch Normalization**:
   - Batch normalization is applied after each convolutional layer (bn3 and bn4) to stabilize and accelerate the training process.
   
3. **Additional Fully Connected Layer**:
   - An extra fully connected layer (fc3) is added to enhance the model's capacity.
   
4. **Dropout Regularization**:
   - Dropout is applied after each fully connected layer (fc1, fc2, and fc3) to prevent overfitting and improve generalization.
   
5. **Padding in Convolutional Layers**:
   - Padding is introduced to convolutional layers to maintain spatial dimensions after convolution.

## Speculation about Suboptimal Performance

- The deeper architecture and increased number of parameters in LeNet4 might have led to overfitting, resulting in worse performance compared to LeNet3.
  
- The added layers in LeNet4 could have increased computational complexity, leading to longer training times and potentially slower convergence.

- Hyperparameter tuning might not have been optimized for LeNet4, causing suboptimal performance compared to LeNet3.
</details>

![image](https://media.github.iu.edu/user/24648/files/71639e82-fed4-444b-a62b-cab39073ef57)


<details>
  <summary>SimpleCNN Architecture Overview</summary>
  
  - **Convolutional Layers (conv1, conv2, conv3)**:
      - Purpose: Extract features from input images using convolution operations.
      - Effect: Multiple convolutional layers are stacked to capture hierarchical features of increasing complexity.

- **Pooling Layer (pool)**:
  - Purpose: Downsample feature maps to reduce spatial dimensions and computational complexity.
  - Effect: Max pooling is used to retain important features while reducing the size of feature maps.

- **Fully Connected Layers (fc1, fc2, fc3)**:
  - Purpose: Perform classification based on extracted features.
  - Effect: Fully connected layers are employed for feature aggregation and classification.

- **Activation Function (ReLU)**:
  - Purpose: Introduce non-linearity to the network.
  - Effect: ReLU activation is chosen for its simplicity and effectiveness in overcoming the vanishing gradient problem.

- **Dropout Layer (dropout)**:
  - Purpose: Regularize the model to prevent overfitting.
  - Effect: Dropout is applied to prevent reliance on specific features and promote robustness.

### Thought Process Behind Using SimpleCNN Model

1. **Simplicity and Efficiency**:
   - Chosen for its straightforward architecture, facilitating ease of implementation and computational efficiency.

2. **Moderate Complexity**:
   - Offers sufficient model capacity for capturing essential features from input images without excessive complexity.

3. **Regularization Mechanisms**:
   - Incorporates dropout regularization to prevent overfitting and improve generalization performance, ensuring robustness in classification tasks.
  
  ![SimpleCNN_20epoch](https://media.github.iu.edu/user/24623/files/fd035bf3-9cb0-4e2b-b270-2d31f5dc47d0)

</details>    


| Model Name | Test Accuracy | Test Loss | Test Accuracy 16x16| Test Loss 16x16|Test Accuracy 8x8| Test Loss 8x8|
|------------|---------------|-----------|---------------------|-----------------|------------------|-----------------|
| LeNet1    | 45.93         | 1.488     | 16.99 | 2.92 | 17.66 | 2.61 |
| LeNet2    | 54.7          | 1.324     | 17.32 | 2.249 | 14.68 | 2.278 |
| LeNet3    | 67.94         | 0.985   | 16.0 | 2.76 | 15.31 | 2.613 |
| LeNet3 (20 epochs)| 70.32 | 0.962 | 15.81 | 2.802 | 15.07 | 2.894 |
| LeNet4 | 45.88 | 1.423 | 17.52 | 2.359 | 16.59 | 2.384 |
| SimpleCNN | 52.700 | 1.329 | 17.49 | 2.666 | 18.44 | 2.586 |
| Vgg      | 76.170.       | 1.643 |24.81 | 5.1   | 35.72  | 3.01 |
|ViT|62.960|1.126|35.58|2.1|31.67|2.2|

| Model Name | Architecture |
|------------|--------------|
| LeNet1    | LeNet1<br>(conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))<br>(conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))<br>(fc1): Linear(in_features=400, out_features=120, bias=True)<br>(fc2): Linear(in_features=120, out_features=84, bias=True)<br>(fc3): Linear(in_features=84, out_features=10, bias=True)<br>(flatten): Flatten(start_dim=1, end_dim=-1)<br>)<br>          |
| LeNet2    | LeNet2(<br>(conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1))<br>(bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>(conv2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))<br>(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,track_running_stats=True)<br>(fc1): Linear(in_features=1600, out_features=512, bias=True)<br>(fc2): Linear(in_features=512, out_features=256, bias=True)<br>(fc3): Linear(in_features=256, out_features=10, bias=True)<br>(flatten): Flatten(start_dim=1, end_dim=-1)<br>(dropout): Dropout(p=0.5, inplace=False)<br>) |
| LeNet3    | LeNet3(<br>(conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1))<br>(bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>(conv2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))<br>(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>(conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))<br>(bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>(avgpool): AdaptiveAvgPool2d(output_size=(1, 1))<br>(fc1): Linear(in_features=128, out_features=512, bias=True)<br>(fc2): Linear(in_features=512, out_features=256, bias=True)<br>(fc3): Linear(in_features=256, out_features=10, bias=True)<br>(flatten): Flatten(start_dim=1, end_dim=-1 ) <br>)|
| LeNet4 | LeNet4(<br>(conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))<br>(bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>(conv2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))<br>(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>(conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>(conv4): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(bn4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>(avgpool): AdaptiveAvgPool2d(output_size=(1, 1))<br>(fc1): Linear(in_features=256, out_features=512, bias=True)<br>(fc2): Linear(in_features=512, out_features=256, bias=True)<br>(fc3): Linear(in_features=256, out_features=128, bias=True)<br>(fc4): Linear(in_features=128, out_features=10, bias=True)<br>(flatten): Flatten(start_dim=1, end_dim=-1)<br>(dropout): Dropout(p=0.5, inplace=False) |
| SimpleCNN | SimpleCNN(<br>(conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)<br>(fc1): Linear(in_features=2048, out_features=512, bias=True)<br>(fc2): Linear(in_features=512, out_features=256, bias=True)<br>(fc3): Linear(in_features=256, out_features=10, bias=True)<br>(relu): ReLU()<br>(dropout): Dropout(p=0.5, inplace=False) |
| Vgg      | VGG(<br>(features): Sequential(<br>(0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(1): ReLU(inplace=True)<br>(2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(3): ReLU(inplace=True)<br>(4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)<br>(5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(6): ReLU(inplace=True)<br>(7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(8): ReLU(inplace=True)<br>(9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)<br>(10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(11): ReLU(inplace=True)<br>(12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(13): ReLU(inplace=True)<br>(14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(15): ReLU(inplace=True)<br>(16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)<br>(17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(18): ReLU(inplace=True)<br>(19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(20): ReLU(inplace=True)<br>(21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(22): ReLU(inplace=True)<br>(23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)<br>(24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(25): ReLU(inplace=True)<br>(26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(27): ReLU(inplace=True)<br>(28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>(29): ReLU(inplace=True)<br>(30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)<br>)<br>(avgpool): AdaptiveAvgPool2d(output_size=(7, 7))<br>(classifier): Sequential(<br>(0): Linear(in_features=25088, out_features=4096, bias=True)<br>(1): ReLU(inplace=True)<br>(2): Dropout(p=0.5, inplace=False)<br>(3): Linear(in_features=4096, out_features=4096, bias=True)<br>(4): ReLU(inplace=True)<br>(5): Dropout(p=0.5, inplace=False)<br>(6): Linear(in_features=4096, out_features=10, bias=True)<br>)<br>)<br>. |
| VisionTransformer |<br>(patch_embedding): Conv2d(3, 128, kernel_size=(4, 4), stride=(4, 4))<br>(transformer_encoder): TransformerEncoder(num_layers=6, TransformerEncoderLayer(<br>&nbsp;&nbsp;&nbsp;&nbsp;(self_attn): MultiheadAttention(out_features=128, num_heads=8)<br>&nbsp;&nbsp;&nbsp;&nbsp;(linear1): Linear(in_features=128, out_features=512)<br>&nbsp;&nbsp;&nbsp;&nbsp;(linear2): Linear(in_features=512, out_features=128)<br>))<br>(fc): Linear(in_features=128, out_features=10)<br>|

### Other Models tried but did not work out : 

#### ResNet :

It is one of the most popular and efficient models for running deep neural networks.

'''

'''

#### HybridCNNViT model :

CNN is a model which can be used for extracting the local features in the images.
And ViT model is pretty good at performing image classification.
The hybrid model is something which we thought would be pretty good for carrying out with the task.
Challenge : Not able to resolve compatibility issues in the architecture.



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
