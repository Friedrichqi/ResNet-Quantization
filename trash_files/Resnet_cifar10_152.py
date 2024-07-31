import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
import os
import os.path as osp
from tqdm import tqdm
import sys
sys.stdout = open("eval_resnet152_cifar10.out", "w")

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained ResNet-50 model
weights = ResNet152_Weights.DEFAULT
model = resnet152(weights=weights)

# Modify the final fully connected layer to match the number of classes in CIFAR-10 (10 classes)
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Define the preprocessing pipeline for CIFAR-10
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Paths to the CIFAR-10 dataset
data_dir = '/data/home/qyjh/Quantization_evaluation/cifar-10'

# Load the CIFAR-10 training and test datasets
train_dataset = datasets.CIFAR10(
    root=data_dir, train=True, download=True, transform=preprocess
)

test_dataset = datasets.CIFAR10(
    root=data_dir, train=False, download=True, transform=preprocess
)

# Create DataLoaders for the training and test sets
train_loader = DataLoader(
    train_dataset,
    batch_size=64,  # Adjust the batch size according to your GPU memory
    shuffle=True,
    num_workers=4  # Adjust the number of worker processes based on your CPU cores
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,  # Adjust the batch size according to your GPU memory
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
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the weights
            
            running_loss += loss.item()
            progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
        with torch.no_grad():
            accuracy = compute_accuracy(model, test_loader, device)
            print(f'Epoch{epoch+1}: Accuracy on CIFAR-10 test set: {accuracy * 100:.2f}%')

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
torch.save(model.state_dict(), 'model/Resnet152_CIFAR10.pt')

# Compute the accuracy on the test set
accuracy = compute_accuracy(model, test_loader, device)
print(f'Accuracy on CIFAR-10 test set: {accuracy * 100:.2f}%')
