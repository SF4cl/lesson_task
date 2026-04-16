import json
import os
import random
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset


def read_split_data(root: str, val_rate: float = 0.2):
    """
    按类别读取图像路径，并按比例拆分训练集/验证集。

    Returns:
        train_images_path, train_images_label, val_images_path, val_images_label
    """
    random.seed(0)
    assert os.path.exists(root), f"dataset root: {root} does not exist."

    # 找到类别文件夹，如 ["cow", "sheep"]
    all_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    all_class.sort()

    # 类别到索引映射，如 {"cow": 0, "sheep": 1}
    class_indices = dict((k, v) for v, k in enumerate(all_class))

    # 保存反向映射到 json：{"0": "cow", "1": "sheep"}
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open("class_indices.json", "w", encoding="utf-8") as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []

    supported = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]

    for cla in all_class:
        cla_path = os.path.join(root, cla)
        images = [
            os.path.join(root, cla, i)
            for i in os.listdir(cla_path)
            if os.path.splitext(i)[-1] in supported
        ]
        images.sort()

        image_class = class_indices[cla]
        every_class_num.append(len(images))

        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print(f"{sum(every_class_num)} images were found in the dataset.")
    print(f"{len(train_images_path)} images for training.")
    print(f"{len(val_images_path)} images for validation.")

    return train_images_path, train_images_label, val_images_path, val_images_label


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: List[str], images_class: List[int], transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert("RGB")
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels