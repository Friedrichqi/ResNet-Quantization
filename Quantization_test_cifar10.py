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
from qtorch import FixedPoint
from qtorch.quant import Quantizer, quantizer


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 10
cifar_datasets = {
    10: datasets.CIFAR10,
    100: datasets.CIFAR100
}

# Function to evaluate model accuracy
def evaluate(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, total=len(dataloader))
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_description(f"Accuracy: {100 * correct / total:.2f}%")
    return 100 * correct / total


# Load test dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Paths to the CIFAR-10 dataset
dataset_class = cifar_datasets[num_classes]
data_dir = f'/data/home/qyjh/Quantization_evaluation/cifar-{num_classes}'
test_dataset = dataset_class(
    root=data_dir, train=False, download=True, transform=transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=512,  # Adjust the batch size according to your GPU memory
    shuffle=False,
    num_workers=4  # Adjust the number of worker processes based on your CPU cores
)


resnet_models = [resnet18, resnet34, resnet50, resnet101, resnet152]
idx_list = [18, 34, 50, 101, 152]

for idx, resnet_model in enumerate(resnet_models):
    sys.stdout = open(f"ptq_resnet{idx_list[idx]}_cifar{num_classes}_conv_layerwise.out", "w")

    model = resnet_model()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(f'/data/home/qyjh/Quantization_evaluation/Resnet_Test/model/Resnet{idx_list[idx]}_CIFAR{num_classes}.pt')
    model.load_state_dict(state_dict)
    model = model.to(device)

    model.eval()
    accuracy_original = evaluate(model, test_loader)
    print(f'Accuracy of the original FP32 model: {accuracy_original:.2f}%')
    
    layer_list = [model.layer1, model.layer2, model.layer3, model.layer4]
    for idx_layer, layer in enumerate(layer_list):
        for bit in range(8, 0, -1):
            model.load_state_dict(state_dict)
            model = model.to(device)
            
            weight_quant = FixedPoint(wl=bit, fl=bit>>1 if bit>>1 > 1 else 1)
            weight_quantizer = Quantizer(forward_number=weight_quant, forward_rounding="stochastic")
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    module.weight.data = weight_quantizer(module.weight.data)
                    pass
            
            # Evaluate quantized model
            model.eval()
            accuracy = evaluate(model, test_loader)
            print(f'Accuracy of the conv layer{idx_layer+1} quantized model ({bit}-bit): {accuracy:.2f}%')
            torch.save(model.state_dict(), f'qmodel/Resnet{idx_list[idx]}_CIFAR{num_classes}_layer{idx_layer+1}_conv_{bit}.pt')
                