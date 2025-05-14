import os
import pandas as pd
import re
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm

class SimpleFontDataset(Dataset):
    def __init__(self, image_dir, csv_path, target_tagword, transform=None):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.target_tagword = target_tagword
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.font2label = {}
        grouped = self.df.groupby("fontName")
        for font_name, group in grouped:
            tagwords = set(group["tagWord"].values)
            self.font2label[font_name] = int(self.target_tagword in tagwords)

        # 从图像文件中提取 fontName，并匹配标签
        self.samples = []
        for fname in os.listdir(self.image_dir):
            if not fname.lower().endswith(".png"):
                continue
            match = re.match(r"(.+?)(?:_.*)?\.png", fname)
            if match:
                font_base = match.group(1)
                if font_base in self.font2label:
                    full_path = os.path.join(self.image_dir, fname)
                    self.samples.append((full_path, self.font2label[font_base]))
                    #print(f"✅ 匹配成功: {full_path} -> {self.font2label[font_base]}")


    def __len__(self):
        return len(self.samples)

    '''
    def __getitem__(self, idx):
        font_name, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, f"{font_name}.png")
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label
    '''
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]  # img_path 已经是完整路径
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label


def collect_fontName_tagWord(taglabel_dir, output_csv="fontName_tagWord.csv"):
    data = []
    for filename in tqdm(os.listdir(taglabel_dir)):
        path = os.path.join(taglabel_dir, filename)
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    line = f.readline()
                    tag_words = line.strip().split()
                    for tag in tag_words:
                        data.append([filename, tag])
            except Exception as e:
                print(f"❌ 读取失败: {filename}, 错误: {e}")

    df = pd.DataFrame(data, columns=["fontName", "tagWord"])
    df.to_csv(output_csv, index=False)
    print(f"✅ 成功保存标签文件至: {output_csv}")
