import torch
import torch.nn.functional as F
import torchvision
import os
from torch.utils.data import DataLoader

from vit import ViT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.mnist.MNIST(root=os.path.join('.', 'data'), train=True, transform=trans, download=False)
test_dataset = torchvision.datasets.mnist.MNIST(root=os.path.join('.', 'data'), train=False, transform=trans, download=False)

BATCH_SIZE = 64
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)

model = ViT(image_size=28, patch_size=4, in_channels=1, embedding_size=16, num_classes=10).to(DEVICE)
model.train()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

EPOCH = 50

for epoch in range(EPOCH):
    total_loss = 0
    iters = 0
    for features, labels in train_dataloader:
        logits = model(features.to(DEVICE))
        loss = F.cross_entropy(logits, labels.to(DEVICE))
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iters += 1
    print(f'epoch {epoch}: {total_loss / iters}')
    torch.save(model.state_dict(), f'model{epoch}.pth')



