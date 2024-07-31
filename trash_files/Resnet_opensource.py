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
import sys
sys.stdout = open("eval_opensource.out", "w")

warnings.filterwarnings("ignore")

def load_data(dataset_name, data_dir=None):
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    elif dataset_name == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if data_dir is None:
            raise ValueError("Please provide the path to the ImageNet dataset directory.")
        train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
        test_dataset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
        trainloader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
        testloader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

    else:
        raise ValueError("Invalid dataset name. Choose from 'cifar10' or 'imagenet'.")

    return trainloader, testloader

def create_resnet(model_name, num_classes=200):
    if model_name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    elif model_name == 'resnet101':
        weights = ResNet101_Weights.DEFAULT
        model = resnet101(weights=weights)
    elif model_name == 'resnet152':
        weights = ResNet152_Weights.DEFAULT
        model = resnet152(weights=weights)
    else:
        raise ValueError("Invalid model name. Choose from 'resnet50', 'resnet101', or 'resnet152'.")

    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def train_model(model, train_loader, test_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs for training.")
        model = nn.DataParallel(model)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    test_predictions = []
    test_labels = []
    inference_times = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, unit='batch', disable=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        epoch_train_loss = train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        train_acc = calculate_accuracy(model, train_loader, device)
        train_accuracies.append(train_acc)

        # Evaluation on the test set
        model.eval()
        test_loss = 0.0
        test_preds = []
        test_lbls = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                start_time = time.time()
                outputs = model(inputs)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_lbls.extend(labels.cpu().numpy())

            test_predictions.append(test_preds)
            test_labels.append(test_lbls)

        epoch_test_loss = test_loss / len(test_loader)
        test_losses.append(epoch_test_loss)

        test_acc = accuracy_score(test_lbls, test_preds)
        test_accuracies.append(test_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

    test_predictions = [item for sublist in test_predictions for item in sublist]
    test_labels = [item for sublist in test_labels for item in sublist]

    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, average='weighted')
    recall = recall_score(test_labels, test_predictions, average='weighted')
    f1 = f1_score(test_labels, test_predictions, average='weighted')
    cm = confusion_matrix(test_labels, test_predictions)

    return train_losses, train_accuracies, test_losses, test_accuracies, test_predictions, test_labels, accuracy, precision, recall, f1, cm, inference_times


def calculate_accuracy(model, data_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def plot_losses_accuracies(train_losses, train_accuracies, test_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Plotting training metrics
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r', label='Training loss')
    plt.plot(epochs, test_losses, 'g', label='Test loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.xticks(epochs)  # Set ticks at integer epochs
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    plt.plot(epochs, test_accuracies, 'm', label='Test accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.xticks(epochs)  # Set ticks at integer epochs
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def main(model_name, dataset_name):
    # Configuration
    data_dir = "/data/home/qyjh/Quantization_evaluation/tiny-imagenet-200"  # Path to the ImageNet dataset directory (required if dataset_name is 'imagenet')
    num_epochs = 10
    num_classes = 10 if dataset_name == 'cifar10' else 200  # Number of classes in the dataset

    # Load data
    train_loader, test_loader = load_data(dataset_name, data_dir)

    # Create model
    model = create_resnet(model_name, num_classes=num_classes)

    # Train model
    train_losses, train_accuracies, test_losses, test_accuracies, test_predictions, test_labels, accuracy, precision, recall, f1, cm, inference_times = train_model(model, train_loader, test_loader, num_epochs=num_epochs)

    # Print average inference time
    average_inference_time = sum(inference_times) / len(inference_times)
    print("Average Inference Time:", average_inference_time)

    # Plot training and testing metrics
    plot_losses_accuracies(train_losses, train_accuracies, test_losses, test_accuracies)

    # Print detailed statistics or visualize results further
    print("Train Losses:", train_losses)
    print("Train Accuracies:", train_accuracies)
    print("Val Losses:", test_losses)
    print("Val Accuracies:", test_accuracies)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)
    
    if not osp.exists('model'):
        os.makedirs('model')
        torch.save(model.state_dict(), 'model/Resnet50_opensource.pt')

if __name__ == '__main__':
    model = "resnet50" # Choose from 'resnet50', 'resnet101', or 'resnet152'
    dataset = "imagenet" # Choose from 'cifar10' or 'imagenet'
    main(model,dataset)