import os
import time
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from logic_cnn_vision.train_vision_logic_cnn import LocalizerLogicCNN, parse_filename_bbox

IMAGE_SIZE = 28


def load_image_and_gt(folder, n):
    files = sorted([p for p in Path(folder).glob('*.png')])
    if len(files) == 0:
        raise ValueError('Dataset folder is empty')
    idx = n % len(files)
    p = files[idx]
    img = Image.open(p).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    X = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    gt = parse_filename_bbox(p.name)
    return X, gt, p


def visualize_prediction(img_array, pred_box, gt_box=None, show=True):
    fig, ax = plt.subplots(1)
    ax.imshow(img_array.squeeze(), cmap='gray', vmin=0, vmax=1)
    px = pred_box[0]
    py = pred_box[1]
    ph = pred_box[2]
    pw = pred_box[3]
    x = px * (IMAGE_SIZE - 1)
    y = py * (IMAGE_SIZE - 1)
    h = ph * (IMAGE_SIZE - 1)
    w = pw * (IMAGE_SIZE - 1)
    rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    if gt_box is not None:
        gx, gy, gh, gw = gt_box
        rect2 = patches.Rectangle((gx, gy), gw, gh, linewidth=1.0, edgecolor='g', facecolor='none')
        ax.add_patch(rect2)
    ax.set_title('Red=predicted, Green=ground-truth')
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='localization-dataset')
    parser.add_argument('--n', type=int, default=0)
    parser.add_argument('--model', type=str, default='localizer_logic_model.pth')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    X, gt, p = load_image_and_gt(args.dataset_dir, args.n)
    print('Loaded image:', p)
    print('Ground truth bbox (x,y,h,w pixels):', gt)

    device = torch.device(args.device)
    model = LocalizerLogicCNN()
    model.load_state_dict(torch.load(args.model))
    model.to(device)
    model.eval()

    with torch.no_grad():
        pred = model(X.to(device))        
    pred_box = pred[0].cpu().numpy()
    print('Predicted (normalized x,y,h,w):', pred_box)
    visualize_prediction(X[0], pred_box, gt_box=gt)


if __name__ == '__main__':
    main()
