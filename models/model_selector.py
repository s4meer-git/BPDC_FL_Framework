from models.cnn_mnist import CNNMNIST
from models.cnn_cifar10 import CNNCIFAR10
from models.resnet_tiny import TinyImageNetResNet18


# Receives the dataset name from main.py and returns the matching MNIST, CIFAR-10 or Tiny ImageNet model.
def build_model(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        return CNNMNIST()

    if dataset_name == "cifar10":
        return CNNCIFAR10()

    if dataset_name == "tinyimagenet":
        return TinyImageNetResNet18(200)

    raise ValueError("Unknown dataset name: " + dataset_name)
