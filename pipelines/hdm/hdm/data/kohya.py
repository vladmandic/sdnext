import os
import sys
import io
import math
import random
import pickle
import tempfile
from collections import defaultdict

import torch
import torchvision.transforms as transforms
import torch.utils.data as Data
import numpy as np
import imagesize
from tqdm import tqdm
from PIL import Image

from xut.modules.axial_rope import make_cropped_pos, make_axial_pos_no_cache


def get_files(folder):
    if os.path.isdir(folder):
        return [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if any(f.endswith(ext) for ext in [".jpg", ".png", ".jpeg", ".webp"])
        ]
    else:
        return None


def load_npy(path):
    with open(path, "rb") as f:
        raw_data = f.read()
    if sys.platform == "win32":
        data = np.load(io.BytesIO(raw_data))
    else:
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(raw_data)
            tmp.flush()
            data = np.load(tmp.name, mmap_mode="r")
    return data


def load_pickle(path):
    with open(path, "rb") as f:
        raw_data = f.read()
    data = pickle.loads(raw_data)
    return data


def conver_rgb(x):
    return x.convert("RGB")


class KohyaDataset(Data.Dataset):
    def __init__(
        self,
        size=1024,
        dataset_folder="/mp34-1/danbooru2023",
        transform=None,
        keep_token_seperator="|||",
        tag_seperator="$$",
        seperator=", ",
        group_seperator="%%",
        tag_shuffle=True,
        group_shuffle=True,
        tag_dropout_rate=0.25,
        group_dropout_rate=0.3,
        use_cached_meta=True,
        meta_postfix="_filtered",
    ):
        self.dataset_folder = dataset_folder
        if (
            os.path.isfile(os.path.join(dataset_folder, f"metadata{meta_postfix}.npy"))
            and use_cached_meta
        ):
            self.files = load_npy(
                os.path.join(dataset_folder, f"metadata{meta_postfix}.npy")
            )
        else:
            print("Cached metadata not found, generating...")
            files = []
            for entry in os.listdir(dataset_folder):
                if os.path.isdir(os.path.join(dataset_folder, entry)):
                    files.extend(get_files(os.path.join(dataset_folder, entry)))
                elif any(
                    entry.endswith(ext) for ext in [".jpg", ".png", ".jpeg", ".webp"]
                ):
                    files.append(entry)
            files = [(i, os.path.splitext(i)[0] + ".txt") for i in files]
            self.files = np.array(files)
            np.save(os.path.join(dataset_folder, f"metadata{meta_postfix}.npy"), files)
            print("Cached metadata generated and saved")

        self.keep_token_seperator = keep_token_seperator
        self.tag_seperator = tag_seperator
        self.seperator = seperator
        self.group_seperator = group_seperator
        self.tag_shuffle = tag_shuffle
        self.group_shuffle = group_shuffle
        self.tag_dropout_rate = tag_dropout_rate
        self.group_dropout_rate = group_dropout_rate

        self.size = size
        self.transform = transform or transforms.Compose(
            [
                transforms.Lambda(conver_rgb),
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

    def __len__(self):
        return len(self.files)

    def get_caption(self, txt_file):
        if not os.path.isfile(txt_file):
            return ""
        with open(txt_file, "r", encoding="utf-8") as f:
            caption = f.read()

        if self.keep_token_seperator in caption:
            keep_tokens, rest = caption.split(self.keep_token_seperator)
            keep_tokens = [
                i.strip() for i in keep_tokens.split(self.tag_seperator) if i.strip()
            ]
        else:
            keep_tokens = []
            rest = caption

        groups = [i.strip() for i in rest.split(self.group_seperator) if i.strip()]
        if self.group_shuffle:
            random.shuffle(groups)

        for group in groups:
            tags = [
                i.strip()
                for i in group.split(self.tag_seperator)
                if i.strip() and random.random() > self.tag_dropout_rate
            ]
            if self.tag_shuffle:
                random.shuffle(tags)
            if random.random() > self.group_dropout_rate:
                keep_tokens.extend(tags)

        return self.seperator.join(keep_tokens)

    def get_data_from_files(self, img_file, txt_file, resize=None):
        img_path = os.path.join(self.dataset_folder, img_file)
        txt_path = os.path.join(self.dataset_folder, txt_file)
        caption = self.get_caption(txt_path)

        with Image.open(img_path) as img:
            if resize:
                img = img.resize(resize, Image.Resampling.BICUBIC)
            img_t = self.transform(img)
        return img_t, caption

    def make_cropped_pos(self, img_t, target_h, target_w):
        aspect_ratio = target_w / target_h
        aspect_ratio = math.log(
            aspect_ratio
        )  # so we have a:b and b:a have same abs value
        crop_h, crop_w = 0, 0
        if target_h > target_w:
            crop_h = torch.randint(0, target_h - target_w, (1,)).item()
            img = img_t[:, crop_h : crop_h + target_w, :]
        elif target_h < target_w:
            crop_w = torch.randint(0, target_w - target_h, (1,)).item()
            img = img_t[:, :, crop_w : crop_w + target_h]
        else:
            img = img_t

        return img, make_cropped_pos(crop_h, crop_w, target_h, target_w)

    def _getitem(self, img_file, txt_file):
        img_t, caption = self.get_data_from_files(img_file, txt_file)
        target_h, target_w = img_t.shape[1:3]
        aspect_ratio = target_w / target_h
        img, pos_map = self.make_cropped_pos(img_t, target_h, target_w)

        return img, caption, pos_map, aspect_ratio

    def __getitem__(self, index):
        img_file, txt_file = self.files[index]
        return self._getitem(img_file, txt_file)


if __name__ == "__main__":
    import random

    dataset = KohyaDataset(
        dataset_folder="/mp34-1/danbooru2023",
        keep_token_seperator="|||",
        tag_seperator="$$",
        seperator=", ",
        group_seperator="%%",
        tag_shuffle=True,
        group_shuffle=True,
        tag_dropout_rate=0.0,
        group_dropout_rate=0.0,
        use_cached_meta=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        ),
        use_arb=True,
        arb_config={
            "batch_size": 32,
            "target_res": 1024,
            "res_step": 16,
            "seed": 0,
        },
        meta_postfix="_filtered",
    )

    print(len(dataset.batches))
    k, values = dataset.batches[0]
    print(k)
    for v in values:
        print(v)
