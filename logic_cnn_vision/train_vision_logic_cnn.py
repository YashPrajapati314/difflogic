import os
import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from difflogic import Conv, Logic

IMAGE_SIZE = 28
CHANNELS = 1


def generate_circle_image(x, y, radius, save_path):
    img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), 0)
    draw = ImageDraw.Draw(img)
    cx = x
    cy = y
    left = cx - radius
    top = cy - radius
    right = cx + radius
    bottom = cy + radius
    draw.ellipse([left, top, right, bottom], fill=255)
    img.save(save_path)


def synthesize_dataset(folder, num_images=5000, min_radius=1, max_radius=8, seed=1337):
    random.seed(seed)
    os.makedirs(folder, exist_ok=True)
    for i in range(num_images):
        radius = random.randint(min_radius, max_radius)
        cx = random.randint(radius, IMAGE_SIZE - 1 - radius)
        cy = random.randint(radius, IMAGE_SIZE - 1 - radius)
        x = cx - radius
        y = cy - radius
        height = radius * 2
        width = radius * 2
        filename = f"{x},{y},{height},{width}.png"
        save_path = os.path.join(folder, filename)
        generate_circle_image(cx, cy, radius, save_path)
    print(f"Synthesized {num_images} images into {folder}")


def parse_filename_bbox(fname):
    stem = Path(fname).stem
    parts = stem.split(',')
    if len(parts) != 4:
        raise ValueError(f"Unexpected filename format: {fname}")
    x, y, h, w = map(int, parts)
    return x, y, h, w


class CircleDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted([p for p in Path(folder).glob('*.png')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
        X = torch.from_numpy(arr).unsqueeze(0)
        x, y, h, w = parse_filename_bbox(p.name)
        Y = torch.tensor([
            x / (IMAGE_SIZE - 1),
            y / (IMAGE_SIZE - 1),
            h / (IMAGE_SIZE - 1),
            w / (IMAGE_SIZE - 1)
        ], dtype=torch.float32)
        return X, Y


def yxyx_from_xyhw(box):
    x = box[..., 0]
    y = box[..., 1]
    h = box[..., 2]
    w = box[..., 3]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return torch.stack([y1, x1, y2, x2], axis=-1)


def iou_loss(y_true, y_pred):
    true = yxyx_from_xyhw(y_true)
    pred = yxyx_from_xyhw(y_pred)
    y1 = torch.maximum(true[..., 0], pred[..., 0])
    x1 = torch.maximum(true[..., 1], pred[..., 1])
    y2 = torch.minimum(true[..., 2], pred[..., 2])
    x2 = torch.minimum(true[..., 3], pred[..., 3])
    inter_h = torch.maximum(torch.tensor(0.0), y2 - y1)
    inter_w = torch.maximum(torch.tensor(0.0), x2 - x1)
    inter_area = inter_h * inter_w
    true_area = (true[..., 2] - true[..., 0]) * (true[..., 3] - true[..., 1])
    pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    union = true_area + pred_area - inter_area
    iou = torch.where(union > 0, inter_area / union, torch.tensor(0.0))
    loss = 1.0 - iou
    return torch.mean(loss)


class LocalizerLogicCNN(nn.Module):
    def __init__(self, model_scale=16):
        super().__init__()
        
        self.k = model_scale
        
        # self.model = nn.Sequential(
        #     Conv(in_channels=1, out_channels=self.k, kernel_size=3, padding=1, stride=1, depth=3),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     Conv(in_channels=self.k, out_channels=2 * self.k, kernel_size=3, padding=1, stride=1, depth=3),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     Conv(in_channels=2 * self.k, out_channels=4 * self.k, kernel_size=3, padding=1, stride=1, depth=3),
        #     nn.Flatten(),
        #     Logic(in_dim=7 * 7 * 4 * self.k, out_dim=64),
        #     Logic(in_dim=64, out_dim=32),
        #     nn.Linear(32, 4),
        #     nn.Sigmoid()
        # )
        
        self.model = nn.Sequential(
            Conv(in_channels=1, out_channels=self.k, kernel_size=3, padding=1, stride=1, depth=2),
            nn.BatchNorm2d(self.k),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            Conv(in_channels=self.k, out_channels=2*self.k, kernel_size=3, padding=1, stride=1, depth=2),
            nn.BatchNorm2d(2*self.k),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            Conv(in_channels=2*self.k, out_channels=4*self.k, kernel_size=3, padding=1, stride=1, depth=1),
            nn.BatchNorm2d(4*self.k),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),

            nn.Linear(4*self.k, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.model(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='localization-dataset')
    parser.add_argument('--num', type=int, default=5000, help='how many images to synthesize if folder empty')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--model_out', type=str, default='localizer_logic_model.pth')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    folder = args.dataset_dir
    if not os.path.exists(folder) or len(list(Path(folder).glob('*.png'))) == 0:
        print('Dataset missing or empty — synthesizing dataset...')
        synthesize_dataset(folder, num_images=args.num)

    print('Loading dataset...')
    dataset = CircleDataset(folder)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    print('Building model...')
    device = torch.device(args.device)
    model = LocalizerLogicCNN().to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f'Training for {args.epochs} epochs...')
    for epoch in range(args.epochs):
        for X, Y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            Y_hat = model(X)
            loss = iou_loss(Y, Y_hat)
            loss.backward()
            optimizer.step()

        val_loss = 0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                Y_hat = model(X)
                val_loss += iou_loss(Y, Y_hat).item()
        print(f"Epoch {epoch}: Validation Loss: {val_loss / len(val_loader)}")

    print(f'Saving model to {args.model_out}...')
    torch.save(model.state_dict(), args.model_out)
    print('Done.')


if __name__ == '__main__':
    main()
