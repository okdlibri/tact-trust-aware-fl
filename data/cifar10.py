"""CIFAR-10 loading utilities."""
import torchvision
import torchvision.transforms as T

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(root="./data/raw", augment=True):
    train_tfms = [T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    if augment:
        train_tfms = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] + train_tfms
    train_tf = T.Compose(train_tfms)
    test_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)
    return train_set, test_set
