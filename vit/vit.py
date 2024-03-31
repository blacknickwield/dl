import math
import torch
from torch import nn
from typing import Union
from functools import reduce

class ViT(nn.Module):
    def __init__(
            self,
            image_size: Union[int, tuple],
            patch_size: Union[int, tuple],
            in_channels: int,
            embedding_size: int,
            num_classes: int,
        ):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.patch_area = patch_size ** 2 if isinstance(patch_size, int) else reduce(lambda x, y: x * y, patch_size)
        if isinstance(image_size, int):
            self.patches = math.ceil(image_size / patch_size) ** 2 if isinstance(patch_size, int) else reduce(lambda x, y: x * y, (math.ceil(image_size / e) for e in patch_size))
        else:
            self.patches = reduce(lambda x, y: x * y, (math.ceil(ie / patch_size) for ie in image_size) if isinstance(patch_size, int) else (math.ceil(ie / pe) for ie, pe in zip(image_size, patch_size)))

        self.split = nn.Conv2d(in_channels=in_channels, out_channels=self.patch_area, kernel_size=patch_size, stride=patch_size)
        self.embedding = nn.Linear(in_features=self.patch_area, out_features=embedding_size)
        self.cls_token = nn.Parameter(torch.rand(1, 1, embedding_size))
        self.position_embedding = nn.Parameter(torch.rand(1, self.patches + 1, embedding_size))

        self.encoders = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_size, nhead=2, batch_first=True), num_layers=3)
        self.classifier = nn.Linear(in_features=embedding_size, out_features=num_classes)

    def forward(self, X: torch.Tensor):
        X = self.split(X)
        batch_size, patch_size, _, _ = X.shape
        X = X.view(batch_size, patch_size, -1).permute(0, 2, 1)
        X = self.embedding(X)

        cls_token = self.cls_token.expand(batch_size, 1, self.cls_token.size(-1))
        X = torch.cat((cls_token, X), dim=1)
        X = X + self.position_embedding
        
        y = self.encoders(X)
        y = self.classifier(y[:, 0, :])

        return y


if __name__ == '__main__':
    model = ViT(image_size=(28, 28), patch_size=(4, 4), in_channels=1, embedding_size=16, num_classes=10)
    X = torch.randn((1, 1, 28, 28))
    model(X)