# import os
# import pickle
# from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, Datum
# from dassl.utils import mkdir_if_missing
# import torchvision.datasets as tv_datasets
# from torchvision import transforms

# @DATASET_REGISTRY.register()
# class CIFAR10Custom(DatasetBase):
#     dataset_dir = "cifar10"

#     def __init__(self, cfg):
#         root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
#         self.dataset_dir = os.path.join(root, self.dataset_dir)
#         mkdir_if_missing(self.dataset_dir)

#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),  # CLIP expects 224x224
#             transforms.ToTensor()
#         ])

#         train_set = tv_datasets.CIFAR10(root=self.dataset_dir, train=True, download=True, transform=transform)
#         test_set = tv_datasets.CIFAR10(root=self.dataset_dir, train=False, download=True, transform=transform)

#         def convert_to_dassl(data):
#             items = []
#             for img, label in zip(data.data, data.targets):
#                 items.append(Datum(impath=None, image=img, label=label, classname=data.classes[label]))
#             return items

#         train = convert_to_dassl(train_set)
#         val = train[:len(train)//10]   # small subset for val
#         test = convert_to_dassl(test_set)

#         super().__init__(train_x=train, val=val, test=test)
#         self.classnames = train_set.classes
import os
from torchvision import datasets, transforms
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing
from PIL import Image

@DATASET_REGISTRY.register()
class CIFAR10Custom(DatasetBase):

    dataset_dir = "cifar10"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        mkdir_if_missing(self.dataset_dir)

        # Base datasets (no transforms yet)
        train_base = datasets.CIFAR10(self.dataset_dir, train=True, download=True)
        test_base = datasets.CIFAR10(self.dataset_dir, train=False, download=True)

        # Define transforms for Dassl later
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        # Directory to store CIFAR10 images as files for Dassl
        temp_img_dir = os.path.join(self.dataset_dir, "images_tmp")
        mkdir_if_missing(temp_img_dir)

        def convert_to_dassl(data):
            items = []
            for idx, (img, label) in enumerate(zip(data.data, data.targets)):
                classname = data.classes[label]
                img_path = os.path.join(temp_img_dir, f"{classname}_{idx}.png")
                if not os.path.exists(img_path):
                    Image.fromarray(img).save(img_path)
                items.append(Datum(impath=img_path, label=label, classname=classname))
            return items

        train = convert_to_dassl(train_base)
        test = convert_to_dassl(test_base)

        super().__init__(train_x=train, val=test, test=test)
