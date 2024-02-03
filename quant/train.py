from model import QuantNet
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import os
import torch.nn.functional as F

root = os.path.join('.', 'data')
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_dataset = torchvision.datasets.MNIST(root=root, train=True, transform=transforms, download=False)
test_dataset = torchvision.datasets.MNIST(root=root, train=False, transform=transforms, download=False)
batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

def train(model: nn.Module, train_loader: DataLoader, device):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.03)
    model = model.train()
    model = model.to(device=device)
    nums = len(train_loader.dataset)
    correct = 0
    train_loss = 0
    for batch_index, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = F.cross_entropy(input=outputs, target=labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        # correct += labels.eq(outputs.argmax(axis=-1).view_as(labels)).sum().cpu()
        pred = outputs.argmax(dim=1, keepdim=True)

        correct += pred.eq(labels.view_as(pred)).sum().item()

    print(f'train loss: {train_loss / nums} acc: {(100 * correct / nums):.2f}%')
    print(correct, nums)

def collect(model: nn.Module, train_loader: DataLoader, device: torch.device):
    model.collect()
    model = model.to(device=device)
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

    

if __name__ == '__main__':
    model = QuantNet(bits=8, in_channels=1)
    epochs = 30
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda:0')
    collect(model=model, train_loader=train_loader, device=device)

    model.quant()
    for epoch in range(epochs):
        print(f'epoch: {epoch}')
        train(model=model, train_loader=train_loader, device=device)
        
    