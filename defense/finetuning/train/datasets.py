import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR100


class COCOFlickrDataset(Dataset):
    def __init__(
            self,
            image_dir_path,
            annotations_path,
            transform=None,
            is_flickr=False,
            prefix=None,
    ):
        self.image_dir_path = image_dir_path
        self.annotations = json.load(open(annotations_path))["annotations"]
        self.is_flickr = is_flickr
        self.transform = transform
        self.prefix = prefix

    def __len__(self):
        return len(self.annotations)

    def get_img_path(self, idx):
        if self.is_flickr:
            return f"{self.image_dir_path}/{self.annotations[idx]['image_id']}.jpg"
        else:
            return f"{self.image_dir_path}/{self.prefix}{self.annotations[idx]['image_id']:012d}.jpg"

    def __getitem__(self, idx):
        image = Image.open(self.get_img_path(idx))
        caption = self.annotations[idx]["caption"]
        return self.transform(image), caption


class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet1k dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        # target_label = IMAGENET_1K_CLASS_ID_TO_LABEL[target]
        return sample, target

class CIFAR100Dataset(CIFAR100):
    """Class to represent the CIFAR-100 dataset."""

    def __init__(self, root, train=True, transform=None, download=True):
        # Initialize CIFAR-100 with required arguments
        super().__init__(root=root, train=train, transform=transform, download=download)

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        return sample, target