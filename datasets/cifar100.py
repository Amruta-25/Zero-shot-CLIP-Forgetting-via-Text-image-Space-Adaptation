@DATASET_REGISTRY.register()
class CIFAR100Custom(DatasetBase):
    dataset_dir = "cifar100"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        mkdir_if_missing(self.dataset_dir)

        train_base = datasets.CIFAR100(self.dataset_dir, train=True, download=True)
        test_base = datasets.CIFAR100(self.dataset_dir, train=False, download=True)

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
