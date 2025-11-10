import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

IMAGE_SIZE = 28


def parse_filename_bbox(fname):
    stem = Path(fname).stem
    parts = stem.split(',')
    if len(parts) != 4:
        raise ValueError(f"Unexpected filename format: {fname}")
    x, y, h, w = map(int, parts)
    return x, y, h, w


def load_image_and_gt(folder, n):
    files = sorted([p for p in Path(folder).glob('*.png')])
    if len(files) == 0:
        raise ValueError('Dataset folder is empty')
    idx = n % len(files)
    p = files[idx]
    img = Image.open(p).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    X = arr.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 1))
    gt = parse_filename_bbox(p.name)
    return X, gt, p


def visualize_prediction(img_array, pred_box, gt_box=None, show=True):
    # pred_box in normalized x,y,h,w (0..1). Convert to pixels
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
    # draw predicted bbox (red)
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
    parser.add_argument('--model', type=str, default='localizer_model.h5')
    args = parser.parse_args()

    X, gt, p = load_image_and_gt(args.dataset_dir, args.n)
    print('Loaded image:', p)
    print('Ground truth bbox (x,y,h,w pixels):', gt)

    model = tf.keras.models.load_model(args.model, compile=False)
    pred = model.predict(X)
    pred_box = pred[0]
    print('Predicted (normalized x,y,h,w):', pred_box)
    # visualize
    visualize_prediction(X[0], pred_box, gt_box=gt)


if __name__ == '__main__':
    main()

