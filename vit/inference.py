import torch
import torchvision
import os
from torch.utils.data import DataLoader
from vit import ViT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_dataset = torchvision.datasets.mnist.MNIST(root=os.path.join('.', 'data'), train=False, transform=trans, download=False)

BATCH_SIZE = 64
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

model = ViT(image_size=28, patch_size=4, in_channels=1, embedding_size=16, num_classes=10).to(DEVICE)
model.load_state_dict(torch.load(os.path.join('.', 'weights', 'model49.pth')))
model.eval()

acc = 0
total = len(test_dataloader.dataset)
for features, labels in test_dataloader:
    logits = model(features.to(DEVICE))
    predictions = logits.argmax(-1)
    acc += ((predictions.view_as(labels)) == labels.to(DEVICE)).sum()

print(f'acc: {(100 * acc / total):.2f}%')







