import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet50, resnet101, resnet152
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import time
from sklearn.metrics import accuracy_score
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os.path as osp
import sys
sys.stdout = open("eval.out", "w")

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained ResNet-50 model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Modify the final fully connected layer to match the number of classes in Tiny ImageNet (200 classes)
num_classes = 200
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


# Define the preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Paths to the Tiny ImageNet dataset
train_dir = '/data/home/qyjh/Quantization_evaluation/tiny-imagenet-200/train'
val_dir = '/data/home/qyjh/Quantization_evaluation/tiny-imagenet-200/val'
test_dir = '/data/home/qyjh/Quantization_evaluation/tiny-imagenet-200/test'

# Load the Tiny ImageNet training and validation datasets
train_dataset = datasets.ImageFolder(
    train_dir,
    transform=preprocess
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=preprocess
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=preprocess
)

# Create DataLoaders for the training and validation sets
train_loader = DataLoader(
    train_dataset,
    batch_size=256,  # Adjust the batch size according to your GPU memory
    shuffle=True,
    num_workers=4  # Adjust the number of worker processes based on your CPU cores
)
total_batch = len(train_loader)

val_loader = DataLoader(
    val_dataset,
    batch_size=256,  # Adjust the batch size according to your GPU memory
    shuffle=False,
    num_workers=4  # Adjust the number of worker processes based on your CPU cores
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,  # Adjust the batch size according to your GPU memory
    shuffle=False,
    num_workers=4  # Adjust the number of worker processes based on your CPU cores
)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the weights
            
            running_loss += loss.item()
            print(f"Epoch{epoch}: Progress: {100 * (idx+1) / total_batch}%")
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/total_batch:.4f}')
        
        with torch.no_grad():    
            accuracy = compute_accuracy(model, test_loader, device)
            print(f'Accuracy on Tiny ImageNet test set: {accuracy * 100:.2f}%')

# Function to compute accuracy
def compute_accuracy(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Train the model
num_epochs = 10  # Set the number of epochs for training
train_model(model, train_loader, criterion, optimizer, device, num_epochs)

if not osp.exists('model'):
    os.makedirs('model')
    torch.save(model.state_dict(), 'model/Resnet50.pt')

# Compute the accuracy on the validation set
accuracy = compute_accuracy(model, val_loader, device)
print(f'Accuracy on Tiny ImageNet validation set: {accuracy * 100:.2f}%')
