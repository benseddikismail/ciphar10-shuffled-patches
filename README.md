# B657 Assignment 2: Image transformations and deep learning

### Techniques Implemented
#### Normalization
Normalization is a preprocessing technique used to standardize input data, ensuring that it has a mean of zero and a standard deviation of one. In the context of image data, normalization is commonly applied to each pixel's intensity values to bring them into a common scale. This helps in improving the convergence speed during training and stabilizing the learning process.

#### Early Stopping
Early stopping is a regularization technique used during the training of machine learning models to prevent overfitting. It works by monitoring the model's performance on a validation set during training. If the performance stops improving or starts deteriorating for a specified number of consecutive epochs (known as the patience parameter), training is stopped early to prevent overfitting to the training data.

#### Learning Rate Scheduler
A learning rate scheduler is used to adjust the learning rate during training dynamically. The learning rate determines the step size at each iteration during gradient descent optimization. A scheduler allows the learning rate to be decreased gradually during training, typically when the validation loss plateaus or stops improving. This can help improve convergence and prevent the model from getting stuck in local minima.

## Comparison of Test Accuracy and Loss (3 best models)

## Dataset creation and PCA analysis

We wrote a script to set up a dataset for our image classification task using CIFAR-10. We randomly selected 25 images from the CIFAR-10 test dataset. These images will be the core of our dataset.Then, we decided to add some variety to our dataset using data augmentation. We created shuffled versions of each of those 25 images. We divided each image into patches of two different sizes, 16x16 and 8x8. After that, we shuffled these patches randomly and put them back together to form a new image. 
Finally,We combined them with the original images to make our final dataset. This mix of original and shuffled images adds more diversity to our dataset..


## Transform and Plots

![WhatsApp Image 2024-04-12 at 23 36 43](https://media.github.iu.edu/user/24623/files/3d5f35be-17f4-4846-b531-5fcc0434819a)

Data points within each class appear to cluster together to some extent, indicating similarity among samples within the same class.
There might be a few outliers present in the dataset, visible as data points that are distant from the main clusters.

Higher variance explained by the first few components suggests that they capture most of the variability in the data

## Insights

1. We noticed that both ViT and Convolutional Neural Networks (CNNs) with fully connected layersn achieved better accuracy on shuffled images. 

2. This suggests proficiency in able to capturing global dependencies, likely owing to ViT's self-attention mechanism and CNNs' convolutional operations. 

3. Despite spatial disruptions, they gave higher accuracy by capturing details in the shuffled image classification.

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
  
  ![LeNet1](https://media.github.iu.edu/user/24623/files/3580a31f-8551-48fc-9197-9a2b417e9f6b)

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


#### ResNet :

It is one of the most popular and efficient models for running deep neural networks.
But it took too long to run.

#### HybridCNNViT model :

CNN is a model which can be used for extracting the local features in the images.
And ViT model is pretty good at performing image classification.
The hybrid model (mix of CNN and ViT) is something which we thought would be pretty good for carrying out with the task.
Challenge : Not able to resolve compatibility issues in the architecture.



<!--### Description
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
- l2_regularization (float): L2 regularization strength (default: 0.0). -->


## CIFAR10 Model

### Initial Attempts

In our initial approach to classifying CIFAR-10 images, we employed a sequential Convolutional Neural Network (CNN) architecture. Specifically, the first convolutional layer was configured with 3 input channels and 16 output channels, followed by subsequent layers with 16 and 32 channels, and then 32 and 64 channels, respectively. Channels in a CNN represent the matrices utilized in the dot product operation during feature selection, crucial for identifying patterns in the input data.
Furthermore, Maxpooling layers were used for downsampling the input representation by selecting the maximum value within defined window sizes along each dimension along the features axis. Additionally, a dropout layer with a dropout rate of 50% was incorporated into the model to mitigate overfitting tendencies commonly observed in deep learning models.
Despite these enhancements, our initial model achieved an accuracy of 68.93%. To further improve accuracy, additional optimization of hyperparameters such as learning rate adjustment and fine-tuning of model parameters would be necessary. This iterative process of parameter tuning and experimentation is essential for achieving higher classification accuracy and better model performance on the CIFAR-10 dataset.  

<figure>
   <figcaption><strong>Initial CIFAR10 Model: Training/Validation Loss vs Epochs</strong></figcaption>
  <img
  src="https://github.iu.edu/cs-b657-sp2024/avmandal-ysampath-isbens-a2/blob/main/img/init-ciphar10-loss.png"
  alt="Initial CIFAR10 Model: Training/Validation Loss vs Epochs">
</figure>
 \
<figure>
   <figcaption><strong>Initial CIFAR10 Model: Training/Validation Accuracy vs Epochs</strong></figcaption>
  <img
  src="https://github.iu.edu/cs-b657-sp2024/avmandal-ysampath-isbens-a2/blob/main/img/init-ciphar10-accuracy.png"
  alt="Initial CIFAR10 Model: Training/Validation Accuracy vs Epochs">
</figure>


In our pursuit of classifying CIFAR-10 images, we conducted extensive experiments employing various state-of-the-art models, including VGG, ResNet, DLA, and Vision Transformer (ViT). While these models have demonstrated high accuracy in classifying CIFAR-10 images in previous studies, our experiments revealed that achieving desired accuracies necessitated extensive training durations and significant computational resources. For instance, training these models to reach optimal accuracies required running for over 100 epochs with low learning rates and small batch sizes, resulting in prolonged training times even when using IU GPU machines or cloud computing platforms like Colab. Due to the complexity of these models and resource constraints, we could only afford to train them for a maximum of 20 epochs. Despite these limitations, our experiments yielded promising results, with the VGG model achieving the highest accuracy of 76%, as detailed in the experiments section below.

### Best Performing Model for CIFAR10

The best performing model we have developed for classifying CIFAR-10 images emerged as an evolution of our initial CNN attempt, tailored to accommodate our computational constraints while delivering robust performance. This model features an expanded architecture with additional convolutional blocks characterized by a kernel size of 3 and modified channel sizes. The model architecture comprises a series of convolutional and pooling layers followed by fully connected layers for classification. Specifically, the convolutional layers are organized into a sequential structure, incorporating batch normalization and ReLU activation functions for enhanced training stability and non-linearity. Dropout layers are strategically inserted to mitigate overfitting during training.

The model architecture can be summarized as follows:
```
Net(
  (cnn): CNN(
    (conv_layer): Sequential(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(32)
      (2): ReLU(inplace=True)
      (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): ReLU(inplace=True)
      (5): MaxPool2d(kernel_size=2, stride=2)
      (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): BatchNorm2d(128)
      (8): ReLU(inplace=True)
      (9): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (10): ReLU(inplace=True)
      (11): MaxPool2d(kernel_size=2, stride=2)
      (12): Dropout2d(p=0.05)
      (13): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (14): BatchNorm2d(256)
      (15): ReLU(inplace=True)
      (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (17): ReLU(inplace=True)
      (18): MaxPool2d(kernel_size=2, stride=2)
    )
    (fc_layer): Sequential(
      (0): Dropout(p=0.1)
      (1): Linear(in_features=4096, out_features=1024, bias=True)
      (2): ReLU(inplace=True)
      (3): Linear(in_features=1024, out_features=512, bias=True)
      (4): ReLU(inplace=True)
      (5): Dropout(p=0.1)
      (6): Linear(in_features=512, out_features=10, bias=True)
    )
  )
)
```
This model was trained with a batch size of 64, a learning rate of 0.001, and L2 regularization of 0.001. Remarkably, after training for just 30 epochs, the model exhibited outstanding performance, achieving an impressive test accuracy of **82.5%**.

<figure>
   <figcaption><strong>Best CIFAR10 Model: Training/Validation Loss vs Epochs</strong></figcaption>
  <img
  src="https://github.iu.edu/cs-b657-sp2024/avmandal-ysampath-isbens-a2/blob/main/img/ciphar10-loss.png"
  alt="Best CIFAR10 Model: Training/Validation Loss vs Epochs">
</figure>
\
<figure>
   <figcaption><strong>Best CIFAR10 Model: Training/Validation Accuracy vs Epochs</strong></figcaption>
  <img
  src="https://github.iu.edu/cs-b657-sp2024/avmandal-ysampath-isbens-a2/blob/main/img/ciphar10-accuracy.png"
  alt="Best CIFAR10 Model: Training/Validation Accuracy vs Epochs">
</figure>
<br>
The following table summarizes the results of the 3 most notable tested on ciphar10: 

| Model Name | Test Accuracy | Test Loss | Test Accuracy 16x16| Test Loss 16x16|Test Accuracy 8x8| Test Loss 8x8|
|------------|---------------|-----------|---------------------|-----------------|------------------|-----------------|
| Enhanced CNN    | 82.50         | 0.539     | 17.67 | 3.165 | 16.66 | 3.038 |
| Vgg      | 76.17       | 1.643 |24.81 | 5.1   | 35.72  | 3.01 |
| Initial CNN  | 68.93        | 0.896   | 18.33 | 2.759 | 15.51 | 2.911 |

## D-shuffletruffle Model

The experiments on the 16 x 16 CIFAR-10 images were conducted using the following hyperparameters with the provided command:

python3 main.py --epochs 10 --model_class 'D-shuffletruffle' --batch_size 128 --learning_rate 0.00005 --l2_regularization 0.01

```python
python main.py --epochs 10 --model_class 'D-shuffletruffle' --batch_size 128 --learning_rate 0.00005 --l2_regularization 0.01
```

### Plain CIFAR10 Model
While the the plain CIFAR10 model can achieve accuracies of up to 85% on the regular CIFAR10 images, it exhibits poor performance on CIFAR10 images with shuffled patches.
Despite its success on regular CIFAR10 images, the model's performance significantly drops when presented with images containing shuffled patches. The disparity in performance between regular and shuffled images indicates that the model is not indifferent to such variations in image composition.
The model is a convolutional neural network composed of several convolutional and fully connected layers. However, its architecture is tailored for processing structured image data. Shuffling the patches disrupts the spatial relationships between pixels, which are crucial for the model to learn meaningful representations. Consequently, the model struggles to generalize from shuffled images, leading to decreased performance compared to regular CIFAR10 images.
To address this performance gap, alternative architectures or training strategies better suited to handle spatially invariant features could be explored. Additionally, data augmentation techniques tailored to preserve spatial relationships in shuffled images may also help mitigate the performance difference between regular and shuffled CIFAR10 images.

<figure>
   <figcaption><strong>Plain CIFAR10 Model: Training and Validation Accuracies vs Epochs</strong></figcaption>
  <img
  src="https://github.iu.edu/cs-b657-sp2024/avmandal-ysampath-isbens-a2/blob/main/img/plain.png"
  alt="Plain CIFAR10 Model: Training and Validation Accuracies vs Epochs">
</figure>
\
### D-shuffletruffle Model
#### First Attempt: Self-Attention
The journey to find an optimal model for D_shuffle_truffle spanned various architectures, from VGG to DLA to ResNet (refer to the comparative table under the Experiments section), and even plain ViT. While these models demonstrated respectable performance on regular CIFAR10 images, they faced significant challenges when presented with images containing shuffled patches. This performance discrepancy underscores the models' inability to adapt to the variability introduced by shuffled patches. The spatial relationships crucial for traditional convolutional architectures to learn meaningful representations were disrupted, leading to poor performance.
To address this challenge, a novel architecture was considered, integrating self-attention alongside convolutional layers. The inclusion of self-attention enables the model to capture long-range dependencies and effectively learn to reorder the shuffled patches. After the convolutional and pooling layers, the feature maps are reshaped into a sequence of feature vectors, with each vector representing a spatial location in the feature maps. These reshaped feature maps are then passed through the nn.MultiheadAttention module, facilitating the computation of self-attention for each feature vector based on its similarity with all other feature vectors in the sequence.
However, while this innovative approach showed promise on regular CIFAR10 images, yielding an average accuracy of 70%, its performance on shuffled images remained subpar, averaging around 40%. This discrepancy suggests that while the model can learn to reorder shuffled patches to some extent, it still struggles to generalize effectively in the face of such variability. The underlying challenge lies in the complexity of learning meaningful representations from images with shuffled patches, which require the model to discern and adapt to spatial relationships in a highly dynamic and unpredictable manner. Further research and refinement of architectural designs may be necessary to bridge this performance gap effectively.


<figure>
  <figcaption><strong>Self-Attention: Training and Validation Accuracies vs Epochs</strong></figcaption>
  <img
  src="https://github.iu.edu/cs-b657-sp2024/avmandal-ysampath-isbens-a2/blob/main/img/self-attention.png"
  alt="Self-Attention: Training and Validation Accuracies vs Epochs">
</figure>
\
#### The Right Model: PEViT

PEViT, inspired by the paper "Human-imperceptible, Machine-recognizable Images," stands out as a pioneering model capable of achieving nearly identical performance on both regular CIFAR10 images and images with shuffled patches (peaking at around 55%). The key innovation lies in its permutation-invariant design, which ensures that the model remains indifferent to the shuffling of image patches. This is achieved through a unique architecture that combines the principles of Vision Transformers (ViT) with encryption strategies and reference-based positional embeddings.
In PEViT, the input image is decomposed into a sequence of flattened 2D patches, similar to traditional ViT models. However, instead of directly processing these patches, PEViT applies a shuffling operation to rearrange them. This shuffled patch sequence is then fed into the model, along with a class token, to generate an image representation invariant to the order of patch embeddings. Notably, PEViT removes the learned positional encodings typically used in ViT to maintain permutation invariance.
To preserve the permutation-invariant property while introducing positional information, PEViT adopts a reference-based positional embedding approach. This involves mapping each vectorized image patch to the model dimension and applying reference-based positional encoding, which relies on a learnable reference embedding. By incorporating this positional encoding scheme, PEViT ensures that the order of input vectors remains covariant with the permutation, enabling it to learn effectively from shuffled images while maintaining permutation invariance.
The architecture of PEViT with reference-based positional encoding demonstrates the feasibility of introducing positional embedding without sacrificing permutation invariance. This groundbreaking approach allows PEViT to achieve remarkable performance parity between regular CIFAR10 images and images with shuffled patches, marking a significant advancement in the field of image recognition and encryption.

PEViT's architecture can be summarized as follows:
```
(VisionTransformer): VisionTransformer(
   (patch_embed): PatchEmbed(
   (proj): Conv2d(3, 384, kernel_size=(8, 8), stride=(8, 8))
   )
   (pos_drop): Dropout(p=0.0, inplace=False)
   (blocks): ModuleList(
   (0-7): 8 x Block(
      (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (attn): Attention(
         (qkv): Linear(in_features=384, out_features=1152, bias=False)
         (attn_drop): Dropout(p=0.0, inplace=False)
         (proj): Linear(in_features=384, out_features=384, bias=True)
         (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): Identity()
      (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (mlp): Mlp(
         (fc1): Linear(in_features=384, out_features=1152, bias=True)
         (act): GELU(approximate='none')
         (fc2): Linear(in_features=1152, out_features=384, bias=True)
         (drop): Dropout(p=0.0, inplace=False)
      )
   )
   )
   (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
   (head): Linear(in_features=384, out_features=10, bias=True)
   (ref_encoding): Sequential(
   (0): Linear(in_features=384, out_features=384, bias=True)
   (1): ReLU()
   (2): Linear(in_features=384, out_features=384, bias=True)
   (3): Sigmoid()
   )
)
```

<figure>
  <figcaption><strong>PEViT Architecture</strong></figcaption>
  <img
  src="https://github.iu.edu/cs-b657-sp2024/avmandal-ysampath-isbens-a2/blob/main/img/pevit.png"
  alt="PEViT Architecture">
</figure>
\
<figure>
  <figcaption><strong>PEViT: Training and Validation Accuracies vs Epochs</strong></figcaption>
  <img
  src="https://github.iu.edu/cs-b657-sp2024/avmandal-ysampath-isbens-a2/blob/main/img/pevit%20acc.png"
  alt="PEViT: Training and Validation Accuracies vs Epochs">
</figure>

### Performance Summary

The table below summarizes the average performance of the three most notable models experimented with on the 16 x 16 shuffled task. Through various iterations with different hyperparameters, all models have demonstrated improved performance compared to their initial configurations. These enhancements reflect the iterative nature of model development and optimization, where fine-tuning hyperparameters and adjusting architectural components can lead to significant performance gains.


| Model Name | Test Accuracy | Test Loss | Test Accuracy 16x16| Test Loss 16x16|Test Accuracy 8x8| Test Loss 8x8|
|------------|---------------|-----------|---------------------|-----------------|------------------|-----------------|
| PEViT    | 49.50         | 1.520     | 49.50 | 1.520 | 29.25 | 2.163 |
| Self-Attention with Convolution Layers    | 70.5          | 0.900     | 40.8 | 2.084 | 27.66 | 3.145 |
| Plain Ciphar10 Model with Convolution Layers   | 82.50         | 0.539     | 17.67 | 3.165 | 16.66 | 3.038 |

### Challenges and Future Work
One of the most significant challenges encountered throughout this project was compute resources. Training the models required a substantial amount of time and computational power, often posing limitations on the depth of experimentation. Due to these constraints, most models were only trained for 10 epochs, with hyperparameters optimized for speed rather than accuracy. This suggests that the models likely have untapped potential in accurately classifying CIFAR-10 images, given more extensive training.
Notably, PEViT stands out as a model with the potential for significantly higher accuracy, potentially surpassing the 55% mark achieved in limited training runs. However, due to the compute limitations, it was not possible to fully explore its capabilities. Since Vision Transformer (ViT) models typically benefit from extensive training data and epochs, future work could focus on training PEViT for a more extended period, possibly up to 200 epochs.
Moreover, future investigations could involve optimizing hyperparameters similar to those used in the paper for the Data-Efficient Image Transformer (DeiT). Parameters such as a smaller batch size of 32 and a lower learning rate of 0.00005 may be crucial for achieving optimal performance in training PEViT on CIFAR-10 images with shuffled patches. By addressing these challenges and refining the training process, we can further explore the capabilities and invariance of PEViT on shuffled images, potentially uncovering its true potential in image classification tasks.

## D-shuffletruffle Model
