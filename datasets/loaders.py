import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class TinyImageNetValidation(Dataset):


    # Receives the Tiny ImageNet folder and image transform from load_tinyimagenet() to read validation paths and labels.
    def __init__(self, root_directory, transform=None):
        self.root_directory = root_directory
        self.transform = transform
        self.samples = []

        words_file = os.path.join(root_directory, "wnids.txt")
        validation_file = os.path.join(root_directory, "val", "val_annotations.txt")
        validation_images = os.path.join(root_directory, "val", "images")


        # Match the alphabetical class order used by ImageFolder for training.
        with open(words_file, "r") as file:
            class_names = sorted(line.strip() for line in file)

        self.class_to_index = {}
        for index, class_name in enumerate(class_names):
            self.class_to_index[class_name] = index

        with open(validation_file, "r") as file:
            for line in file:
                parts = line.strip().split('\t')
                image_name = parts[0]
                class_name = parts[1]
                image_path = os.path.join(validation_images, image_name)
                class_index = self.class_to_index[class_name]
                self.samples.append((image_path, class_index))


    # Returns the number of validation samples when DataLoader or main.py asks for the dataset size.
    def __len__(self):
        return len(self.samples)


    # Receives a sample index from DataLoader, opens its validation image and returns the transformed image and label.
    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# Receives a loaded train or test dataset from create_clients() and returns its labels for partitioning.
def get_dataset_labels(dataset):
    if hasattr(dataset, "targets"):
        labels = dataset.targets

        if torch.is_tensor(labels):
            labels = labels.tolist()

        return list(labels)

    if hasattr(dataset, "samples"):
        return [sample[1] for sample in dataset.samples]

    return []


# Receives a data folder from load_dataset() and returns the MNIST train/test datasets with their transforms.
def load_mnist(data_directory="data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=data_directory,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_directory,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


# Receives a data folder from load_dataset() and returns the CIFAR-10 train/test datasets with their transforms.
def load_cifar10(data_directory="data"):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root=data_directory,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_directory,
        train=False,
        download=True,
        transform=test_transform
    )

    return train_dataset, test_dataset


# Receives the Tiny ImageNet folder from load_dataset() and builds its training and validation datasets.
def load_tinyimagenet(data_directory="data/tiny-imagenet-200"):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4802, 0.4481, 0.3975),
            (0.2302, 0.2265, 0.2262)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4802, 0.4481, 0.3975),
            (0.2302, 0.2265, 0.2262)
        )
    ])

    train_directory = os.path.join(data_directory, "train")

    if not os.path.exists(train_directory):
        raise FileNotFoundError(
            "Tiny ImageNet was not found. Put tiny-imagenet-200 inside the data folder."
        )

    train_dataset = datasets.ImageFolder(
        root=train_directory,
        transform=train_transform
    )

    test_dataset = TinyImageNetValidation(
        root_directory=data_directory,
        transform=test_transform
    )

    return train_dataset, test_dataset


# Receives the dataset name and optional data folder from main.py and calls the matching dataset loader.
def load_dataset(dataset_name, data_directory="data"):
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        return load_mnist(data_directory)

    if dataset_name == "cifar10":
        return load_cifar10(data_directory)

    if dataset_name == "tinyimagenet":
        tiny_directory = os.path.join(data_directory, "tiny-imagenet-200")
        return load_tinyimagenet(tiny_directory)

    raise ValueError("Unknown dataset name: " + dataset_name)
