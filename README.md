# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement
In real-world scenarios, images are often corrupted by noise due to factors such as poor lighting conditions, sensor limitations, or transmission errors. This noise degrades image quality and makes it difficult for both humans and machine learning models to interpret the visual information accurately.
The problem is to design and implement a Convolutional Autoencoder using deep learning techniques to automatically remove noise from corrupted images and reconstruct clean versions.
For this task, the MNIST dataset of handwritten digits is used. Artificial noise is added to the images, and the autoencoder is trained to recover the original clean images. The performance of the model is evaluated based on how effectively it reconstructs the denoised images.

## DESIGN STEPS
### STEP 1: 
Problem Understanding and Dataset Selection

### STEP 2: 
 Preprocessing the Dataset
 
### STEP 3: 
Design the Convolutional Autoencoder Architecture

### STEP 4: 
Compile and Train the Model

### STEP 5: 
Evaluate the Model

### STEP 6: 
Visualization and Analysis 

## PROGRAM

### Name: SIBHIRAAJ R

### Register Number: 212224230268
```
# Autoencoder for Image Denoising using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transform: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, 7, 7]
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # [B, 1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Name:SIBHIRAAJ R")
print("Register Number:212224230268")
summary(model, input_size=(1, 28, 28))

def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print("Name:SIBHIRAAJ R")
    print("Register Number:212224230268")

    for epoch in range(epochs):
        running_loss = 0.0

        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            # Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")

def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name:SIBHIRAAJ R")
    print("Register Number:212224230268")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)
```

### OUTPUT

### Model Summary
<img width="693" height="442" alt="image" src="https://github.com/user-attachments/assets/55311970-6a8b-4522-b830-f6a3cd2cafac" />


### Training loss
<img width="399" height="175" alt="image" src="https://github.com/user-attachments/assets/712cf555-b6aa-40f2-b553-af0cf10b2ed5" />

## Original vs Noisy Vs Reconstructed Image
<img width="1760" height="579" alt="image" src="https://github.com/user-attachments/assets/8e5c4688-14ae-4e2c-aeb3-28303b3b610e" />


## RESULT
Therefore, To develop a convolutional autoencoder for image denoising application executed successfully.

