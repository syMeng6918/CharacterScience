
import os
import random
import re
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


__all__ = [
    "SingleLetterFontDataset",
    "MultiLetterFontDataset",
]

FONT_REGEX = re.compile(r"^(.*?)(?:_[A-Za-z]{1,2})?\.png$")

def extract_font_name(file_name: str) -> str:
    m = FONT_REGEX.match(file_name)
    return m.group(1) if m else Path(file_name).stem

class SingleLetterFontDataset(Dataset):
    def __init__(self,
                 image_dir: str,
                 csv_path: str,
                 target_tag: str,
                 transform: T.Compose = None):
        self.image_dir = Path(image_dir)
        self.df = pd.read_csv(csv_path)
        self.target_tag = target_tag
        self.transform = transform or T.Compose([
            T.Grayscale(3),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])

        font_has_tag = (
            self.df.groupby("fontName")["tagWord"]
            .apply(lambda s: target_tag in set(s.values))
            .to_dict()
        )

        self.samples: List[Tuple[Path, int]] = []
        for p in self.image_dir.glob("*.png"):
            font = extract_font_name(p.name)
            if font in font_has_tag:
                self.samples.append((p, int(font_has_tag[font])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        img = self.transform(img)
        return img, label

"""class MultiLetterFontDataset(Dataset):
    def __init__(self,
                 image_dir: str,
                 csv_path: str,
                 target_tag: str,
                 N: int = 4,
                 repeat_per_font: int = 3,
                 transform: T.Compose = None,
                 debug: bool = False,
                 concat: bool = True):
        self.image_dir = Path(image_dir)
        self.N = N
        self.debug = debug
        self.concat = concat
        self.transform = transform or T.Compose([
            T.Grayscale(3),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])

        df = pd.read_csv(csv_path)
        font_has_tag = (
            df.groupby("fontName")["tagWord"]
            .apply(lambda s: target_tag in set(s.values))
            .to_dict()
        )

        font2imgs = {}
        for p in self.image_dir.glob("*.png"):
            font = extract_font_name(p.name)
            font2imgs.setdefault(font, []).append(p)

        self.samples: List[Tuple[List[Path], int]] = []
        for font, img_list in font2imgs.items():
            for _ in range(repeat_per_font):
                self.samples.append((img_list, int(font_has_tag.get(font, False))))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_list, label = self.samples[idx]
        picks = random.sample(img_list, self.N) if len(img_list) >= self.N                 else random.choices(img_list, k=self.N)

        if self.debug:
            print(f"[DEBUG] sample{idx} 拼图组成: {[p.name for p in picks]}")

        imgs = [self.transform(Image.open(p).convert("L")) for p in picks]

        if self.concat:
            concat_img = torch.cat(imgs, dim=2)  # (3, 224, 224*N)
            return concat_img, label
        else:
            imgs_tensor = torch.stack(imgs, dim=0)  # (N, 3, 224, 224)
            return imgs_tensor, label"""
class MultiLetterFontDataset(Dataset):
    def __init__(self,
                 image_dir: str,
                 csv_path: str,
                 target_tag: str,
                 N: int = 4,
                 repeat_per_font: int = 3,
                 transform: T.Compose = None,
                 debug: bool = False,
                 concat: bool = True):
        self.image_dir = Path(image_dir)
        self.N = N
        self.debug = debug
        self.concat = concat
        self.transform = transform or T.Compose([
            T.Grayscale(3),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])

        df = pd.read_csv(csv_path)
        font_has_tag = (
            df.groupby("fontName")["tagWord"]
            .apply(lambda s: target_tag in set(s.values))
            .to_dict()
        )

        font2imgs = {}
        for p in self.image_dir.glob("*.png"):
            font = extract_font_name(p.name)
            font2imgs.setdefault(font, []).append(p)

        self.samples: List[Tuple[List[Path], int]] = []
        for font, img_list in font2imgs.items():
            for _ in range(repeat_per_font):
                self.samples.append((img_list, int(font_has_tag.get(font, False))))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_list, label = self.samples[idx]
        picks = random.sample(img_list, self.N) if len(img_list) >= self.N else random.choices(img_list, k=self.N)

        imgs = [self.transform(Image.open(p).convert("L")) for p in picks]

        if self.concat:
            concat_img = torch.cat(imgs, dim=2)
        else:
            concat_img = torch.stack(imgs, dim=0)

        font_name = picks[0].name.split("_")[0]
        return concat_img, label, font_name
