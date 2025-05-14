
import argparse
import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from dataset_repeat import MultiLetterFontDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from efficientnet_pytorch import EfficientNet
from collections import Counter, defaultdict
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from efficientnet_pytorch import EfficientNet
from torchvision.models import mobilenet_v2, resnet18
import torchvision.utils as vutils
import torchvision.transforms.functional as TF

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_model(name="efficientnet"):
    if name == "efficientnet":
        model = EfficientNet.from_pretrained("efficientnet-b0")
        model._fc = nn.Identity()
        backbone_out = 1280
    elif name == "mobilenet":
        model = mobilenet_v2(pretrained=True)
        model.classifier = nn.Identity()
        backbone_out = 1280
    elif name == "resnet18":
        model = resnet18(pretrained=True)
        model.fc = nn.Identity()
        backbone_out = 512
    else:
        raise ValueError(f"Unsupported model: {name}")

    classifier = nn.Sequential(
        nn.Linear(backbone_out, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256, 1)
    )

    return nn.Sequential(model, classifier)

class TransferEfficientNetB0(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.backbone._fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=1.0, gamma_neg=1.0, eps=1e-4):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.eps = eps

    def forward(self, inputs, targets):
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_sigmoid = torch.clamp(inputs_sigmoid, min=self.eps, max=1 - self.eps)
        targets = targets.float()
        pos_inds = targets == 1
        neg_inds = targets == 0

        pos_loss = -((1 - inputs_sigmoid[pos_inds]) ** self.gamma_pos) * torch.log(inputs_sigmoid[pos_inds] + self.eps)
        neg_loss = -((inputs_sigmoid[neg_inds]) ** self.gamma_neg) * torch.log(1 - inputs_sigmoid[neg_inds] + self.eps)

        loss = torch.cat([pos_loss, neg_loss], dim=0)
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, inputs, targets):
        inputs_sigmoid = torch.sigmoid(inputs)
        targets = targets.float()
        loss_pos = -self.alpha * (1 - inputs_sigmoid) ** self.gamma * targets * torch.log(inputs_sigmoid + self.eps)
        loss_neg = -(1 - self.alpha) * inputs_sigmoid ** self.gamma * (1 - targets) * torch.log(1 - inputs_sigmoid + self.eps)
        return (loss_pos + loss_neg).mean()

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, required=True)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--warmup_start_lr', type=float, default=0.001)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--loss', type=str, default='asl', choices=['asl', 'focal', 'bce'])
parser.add_argument('--model', type=str, default='mobilenet', choices=['mobilenet', 'efficientnet', 'resnet18'])
parser.add_argument('--letters', type=int, default=4)
parser.add_argument('--repeat', type=int, default=3)
args = parser.parse_args()

DATA_DIR = "dataset/fontimage_preprocessed_reference"
LABEL_CSV = "dataset/fontName_tagWord.csv"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = MultiLetterFontDataset(
    image_dir=DATA_DIR,
    csv_path=LABEL_CSV,
    target_tag=args.tag,
    N=args.letters,
    repeat_per_font=args.repeat,
    transform=transform,
    concat=True
)

font_has_tag = pd.read_csv(LABEL_CSV).groupby("fontName")["tagWord"].apply(lambda s: args.tag in set(s.values)).to_dict()
labels = [label for _, label in dataset.samples]
counts = Counter(labels)
weights = [1.0 / counts[label] for label in labels]
sampler = WeightedRandomSampler(weights, len(weights))

train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
subset_indices = train_dataset.indices
train_labels = [labels[i] for i in subset_indices]
train_counts = Counter(train_labels)
train_weights = [1.0 / train_counts[l] for l in train_labels]
train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

print(f"训练集标签分布: {counts}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(name=args.model).to(device)
if args.loss == 'asl':
    criterion = AsymmetricLoss()
elif args.loss == 'bce':
    #pos_weight = torch.tensor([counts[0] / counts[1]]).to(device)
    pos_weight = torch.tensor([train_counts[0] / train_counts[1]]).to(device)
    print(f"🔢 正样本: {train_counts[1]}, 负样本: {train_counts[0]}, pos_weight: {pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    criterion = FocalLoss()

optimizer = optim.AdamW(model.parameters(), lr=args.warmup_start_lr)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
scaler = GradScaler()

best_valid_loss = float('inf')
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(1, args.epochs + 1):
    model.train()
    running_loss = 0.0
    for images, labels,_ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
        images = images.to(device)
        labels = labels.float().clamp(0, 1).view(-1, 1).to(device)
        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print("[ERROR] Loss is NaN or Inf at epoch", epoch)
            exit()

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    model.eval()
    valid_loss = 0.0
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    font_stats = defaultdict(lambda: {'true': None, 'pred0': 0, 'pred1': 0, 'total': 0})

    with torch.no_grad():
        for images, labels, font_names in tqdm(test_loader, desc=f"Epoch {epoch}/{args.epochs} [Valid]"):
            images = images.to(device)
            labels = labels.float().clamp(0, 1).view(-1, 1).to(device)
            outputs = torch.sigmoid(model(images))
            preds = (outputs >= 0.5).float()
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

            for fname, true, pred in zip(font_names, labels.cpu().numpy(), preds.cpu().numpy()):
                fname = fname.split("_")[0]
                font_stats[fname]['true'] = int(true[0])
                font_stats[fname]['total'] += 1
                if pred[0] >= 0.5:
                    font_stats[fname]['pred1'] += 1
                else:
                    font_stats[fname]['pred0'] += 1
            # 保存拼图图像（注意只适用于 batch_size = 1 或逐个保存）
            '''for img_tensor, fontname in zip(images, font_names):
                fname = os.path.basename(fontname)
                save_path = os.path.join("checkpoints/val_images", f"{fname}_epoch{epoch}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # 确保是 [3,H,W] 的 tensor，并转为 PIL 保存
                img = TF.to_pil_image(img_tensor.cpu().float().clamp(0, 1))
                img.save(save_path)
'''
    with open(f"checkpoints/font_prediction_stats_epoch{epoch}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["font_name", "true_label", "total", "predicted_as_1", "predicted_as_0", "always_wrong"])
        for font, stat in font_stats.items():
            # 所有预测都是1 或 都是0，且预测结果全错，才算 always wrong
            always_wrong = int(
                (stat['pred1'] == stat['total'] and stat['true'] == 0) or
                (stat['pred0'] == stat['total'] and stat['true'] == 1)
            )
            writer.writerow([font, stat['true'], stat['total'], stat['pred1'], stat['pred0'], always_wrong])
    print(f"📄 Saved font prediction stats to: checkpoints/font_prediction_stats_epoch{epoch}.csv")
    avg_valid_loss = valid_loss / len(test_loader)
    valid_acc = correct / total

    all_probs_np = np.array(all_probs)
    print(f"📊 Probabilities - Min: {all_probs_np.min():.4f} | Max: {all_probs_np.max():.4f} | Mean: {all_probs_np.mean():.4f}")
    print(f"✅ Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f} | Valid Acc: {valid_acc:.4f}")

    if epoch <= args.warmup_epochs:
        warmup_progress = epoch / args.warmup_epochs
        warmup_lr = args.warmup_start_lr + (args.lr - args.warmup_start_lr) * warmup_progress
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
        print(f"🌟 Warmup阶段，学习率调整为: {warmup_lr:.6f}")
    else:
        cosine_scheduler.step()

    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        torch.save(model.state_dict(), f"checkpoints/{args.tag}_best.pth")
        print(f"✅ 保存当前最优模型到: checkpoints/{args.tag}_best.pth")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"checkpoints/{args.tag}_epoch{epoch}.pth")
        print(f"📦 保存模型到: checkpoints/{args.tag}_epoch{epoch}.pth")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix Epoch {epoch}")
    plt.savefig(f"checkpoints/confmat_epoch{epoch}.png")
    plt.close()

print("🏁 训练结束!")
torch.save(model.state_dict(), f"checkpoints/{args.tag}_final.pth")
print(f"✅ 最终模型保存到: checkpoints/{args.tag}_final.pth")