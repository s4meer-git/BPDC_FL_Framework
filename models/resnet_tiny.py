import torch.nn as nn

from torchvision.models import resnet18


# Receives the class count from build_model() and adjusts ResNet-18 for the 64 by 64 Tiny ImageNet images.
def TinyImageNetResNet18(no_of_classes=200):
    model = resnet18(weights=None)


    # Tiny ImageNet images are 64 x 64, so we use a smaller first convolution.
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )


    # Skip the initial max-pool so the 64 x 64 Tiny ImageNet images do not shrink too early.
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, no_of_classes)

    return model
