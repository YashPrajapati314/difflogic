import os
import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras import layers, models

IMAGE_SIZE = 28
CHANNELS = 1


def generate_circle_image(x, y, radius, save_path):
    # Create black image and draw white circle centered at (cx,cy)
    img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), 0)
    draw = ImageDraw.Draw(img)
    cx = x
    cy = y
    # draw.ellipse expects top-left and bottom-right of bounding box
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
        # ensure circle fully inside image
        cx = random.randint(radius, IMAGE_SIZE - 1 - radius)
        cy = random.randint(radius, IMAGE_SIZE - 1 - radius)
        # bounding box (top-left x,y and height,width). We'll use x,y as top-left corner of bbox
        x = cx - radius
        y = cy - radius
        height = radius * 2
        width = radius * 2
        filename = f"{x},{y},{height},{width}.png"
        save_path = os.path.join(folder, filename)
        generate_circle_image(cx, cy, radius, save_path)
    print(f"Synthesized {num_images} images into {folder}")


def parse_filename_bbox(fname):
    # fname like 'x,y,height,width.png' or without extension
    stem = Path(fname).stem
    parts = stem.split(',')
    if len(parts) != 4:
        raise ValueError(f"Unexpected filename format: {fname}")
    x, y, h, w = map(int, parts)
    return x, y, h, w


def load_dataset(folder):
    files = sorted([p for p in Path(folder).glob('*.png')])
    X = np.zeros((len(files), IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.float32)
    Y = np.zeros((len(files), 4), dtype=np.float32)
    for i, p in enumerate(files):
        img = Image.open(p).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        X[i, :, :, 0] = arr
        x, y, h, w = parse_filename_bbox(p.name)
        # normalize to [0,1] relative to image width/height
        Y[i, 0] = x / (IMAGE_SIZE - 1)
        Y[i, 1] = y / (IMAGE_SIZE - 1)
        Y[i, 2] = h / (IMAGE_SIZE - 1)
        Y[i, 3] = w / (IMAGE_SIZE - 1)
    return X, Y


# IoU computation and loss

def yxyx_from_xyhw(box):
    # input box in xyhw (top-left x,y,h,w) normalized [0,1]
    x = box[..., 0]
    y = box[..., 1]
    h = box[..., 2]
    w = box[..., 3]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return tf.stack([y1, x1, y2, x2], axis=-1)  # return in y1,x1,y2,x2 order


@tf.function
def iou_loss(y_true, y_pred):
    # both y_true, y_pred: batch x 4 normalized
    # Convert to y1,x1,y2,x2
    true = yxyx_from_xyhw(y_true)
    pred = yxyx_from_xyhw(y_pred)
    # intersection
    y1 = tf.maximum(true[..., 0], pred[..., 0])
    x1 = tf.maximum(true[..., 1], pred[..., 1])
    y2 = tf.minimum(true[..., 2], pred[..., 2])
    x2 = tf.minimum(true[..., 3], pred[..., 3])
    inter_h = tf.maximum(0.0, y2 - y1)
    inter_w = tf.maximum(0.0, x2 - x1)
    inter_area = inter_h * inter_w
    true_area = (true[..., 2] - true[..., 0]) * (true[..., 3] - true[..., 1])
    pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    union = true_area + pred_area - inter_area
    iou = tf.where(union > 0, inter_area / union, 0.0)
    # loss = 1 - mean IoU
    loss = 1.0 - iou
    return tf.reduce_mean(loss)


def build_model():
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=iou_loss, metrics=[])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='localization-dataset')
    parser.add_argument('--num', type=int, default=5000, help='how many images to synthesize if folder empty')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--model_out', type=str, default='localizer_model.h5')
    args = parser.parse_args()

    folder = args.dataset_dir
    if not os.path.exists(folder) or len(list(Path(folder).glob('*.png'))) == 0:
        print('Dataset missing or empty — synthesizing dataset...')
        synthesize_dataset(folder, num_images=args.num)

    print('Loading dataset...')
    X, Y = load_dataset(folder)
    # shuffle and split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(0.9 * len(X))
    train_idx = idx[:split]
    val_idx = idx[split:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    print('Building model...')
    model = build_model()
    model.summary()

    print(f'Training for {args.epochs} epochs...')
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=args.epochs, batch_size=args.batch)

    print(f'Saving model to {args.model_out}...')
    model.save(args.model_out)
    print('Done.')


if __name__ == '__main__':
    main()