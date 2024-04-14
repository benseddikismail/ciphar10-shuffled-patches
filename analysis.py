import numpy as np
import torchvision
from einops import rearrange
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
from PIL import Image
from main import Net, Net_D_shuffletruffle, Net_N_shuffletruffle
#

def shuffle(patch_size: int, data_point):
    height_reshape = int(np.shape(data_point)[0] / patch_size)
    img = rearrange(data_point, '(h s1) (w s2) c -> (h w) s1 s2 c', s1=patch_size, s2=patch_size)
    np.random.shuffle(img)
    img = rearrange(img, '(h w) s1 s2 c -> (h s1) (w s2) c', h=height_reshape, s1=patch_size, s2=patch_size)
    return img

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

def create_patch_shuffled_data(dataset: torchvision.datasets, file_name: str, patch_size: int = 16) -> None:
    images = []
    labels = []
    for i, l in dataset:
        m = np.asarray(i)
        m = shuffle(patch_size=patch_size, data_point=m)
        images.append(m)
        labels.append(l)
    np.savez(file=file_name, data=images, labels=labels)

np.random.seed(1)

create_patch_shuffled_data(testset, file_name="test_patch_16", patch_size=16)
create_patch_shuffled_data(testset, file_name="test_patch_8", patch_size=8)

original_images = []
original_labels = []

for i, l in testset:
    original_images.append(i)
    original_labels.append(l)

original_images = np.array(original_images)
original_labels = np.array(original_labels)

unique_indices = np.random.choice(len(original_labels), size=25, replace=False)
selected_images = original_images[unique_indices]
selected_labels = np.arange(1, 26)

label_mapping = dict(zip(selected_labels, selected_labels))

small_dataset_images = []
small_dataset_labels = []

for i in range(75):
    random_index = np.random.randint(len(selected_labels))
    original_img = selected_images[random_index]
    shuffled_16_patch_img = np.load("test_patch_16.npz")["data"][random_index]
    shuffled_8_patch_img = np.load("test_patch_8.npz")["data"][random_index]

    small_dataset_images.extend([original_img, shuffled_16_patch_img, shuffled_8_patch_img])
    small_dataset_labels.extend([selected_labels[random_index]] * 3)

np.savez(file="small_dataset", data=small_dataset_images, labels=small_dataset_labels, label_mapping=label_mapping)

### CIPHAR10 MODEL

small_dataset = np.load("small_dataset.npz", allow_pickle=True)
images = small_dataset["data"]
labels = small_dataset["labels"]

model = Net()
model.load_state_dict(torch.load("model.pth"))
model.eval()

def get_embeddings(model, images):
    embeddings = []
    for img in images:
        img_tensor = torch.unsqueeze(ToTensor()(img), 0)  # Convert image to tensor and add batch dimension
        with torch.no_grad():
            embedding = model.cnn.conv_layer(img_tensor).view(1, -1)  # Reshape to (1, 4096)
        embeddings.append(embedding)
    return embeddings

embeddings = get_embeddings(model, images)

embeddings_flat = [embedding.numpy().flatten() for embedding in embeddings]

pca = PCA(n_components=2)
transformed_embeddings = pca.fit_transform(embeddings_flat)

plt.figure(figsize=(8, 6))
plt.scatter(transformed_embeddings[:, 0], transformed_embeddings[:, 1], c=labels, cmap='tab10')
plt.title('PCA Transformed Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Label')
plt.grid(True)
plt.show()

### Net_D_shuffletruffle

small_dataset = np.load("small_dataset.npz", allow_pickle=True)
images = small_dataset["data"]
labels = small_dataset["labels"]


model = Net_D_shuffletruffle()
model.load_state_dict(torch.load("model_D_shuffletruffle.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to match model input size
    transforms.ToTensor()          # Convert PIL images to tensors
])


def get_embeddings(model, images):
    embeddings = []
    with torch.no_grad():
        for img in images:
            img_tensor = transform(Image.fromarray(img))  
            img_tensor = torch.unsqueeze(img_tensor, 0)    
            embedding = model(img_tensor).numpy()         
            embeddings.append(embedding)
    return embeddings

embeddings = get_embeddings(model, images)

embeddings_flat = np.array(embeddings).reshape(len(images), -1)

pca = PCA(n_components=2)
transformed_embeddings = pca.fit_transform(embeddings_flat)

plt.figure(figsize=(8, 6))
plt.scatter(transformed_embeddings[:, 0], transformed_embeddings[:, 1], c=labels, cmap='tab10')
plt.title('PCA Transformed Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Label')
plt.grid(True)
plt.show()

### Net_N_shuffletruffle

small_dataset = np.load("small_dataset.npz", allow_pickle=True)
images = small_dataset["data"]
labels = small_dataset["labels"]

model = Net_N_shuffletruffle()
model.load_state_dict(torch.load("model_N_shuffletruffle.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to match model input size
    transforms.ToTensor()          # Convert PIL images to tensors
])

def get_embeddings(model, images):
    embeddings = []
    with torch.no_grad():
        for img in images:
            img_tensor = transform(Image.fromarray(img))  # Apply transformations
            img_tensor = torch.unsqueeze(img_tensor, 0)    # Add batch dimension
            embedding = model(img_tensor).numpy()         # Get model output and convert to numpy array
            embeddings.append(embedding)
    return embeddings

embeddings = get_embeddings(model, images)

embeddings_flat = np.array(embeddings).reshape(len(images), -1)

pca = PCA(n_components=2)
transformed_embeddings = pca.fit_transform(embeddings_flat)

plt.figure(figsize=(8, 6))
plt.scatter(transformed_embeddings[:, 0], transformed_embeddings[:, 1], c=labels, cmap='tab10')
plt.title('PCA Transformed Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Label')
plt.grid(True)
plt.show()