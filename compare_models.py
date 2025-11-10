import os
import time
import pandas as pd
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from logic_cnn_vision.train_vision_logic_cnn import LocalizerLogicCNN, parse_filename_bbox
import tensorflow as tf

def load_image_and_gt_lgn(folder, n):
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


def load_image_and_gt_nn(folder, n):
    IMAGE_SIZE = 28
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


import numpy as np

def compute_iou(pred_box, gt_box):
    def to_corners(box):
        cx, cy, w, h = box
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return x1, y1, x2, y2

    x1_p, y1_p, x2_p, y2_p = to_corners(pred_box)
    x1_g, y1_g, x2_g, y2_g = to_corners(gt_box)

    inter_x1 = max(x1_p, x1_g)
    inter_y1 = max(y1_p, y1_g)
    inter_x2 = min(x2_p, x2_g)
    inter_y2 = min(y2_p, y2_g)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_pred = (x2_p - x1_p) * (y2_p - y1_p)
    area_gt = (x2_g - x1_g) * (y2_g - y1_g)
    union = area_pred + area_gt - intersection

    iou = intersection / union if union > 0 else 0.0
    return iou



def compare_performance():
    
    dataset_dir = 'localization-dataset'
    device = 'cuda'
    IMG_DIM = 28
    NUMBER_OF_IMAGES = 100
    
    lgn_model_name = 'new_localizer_logic_model.pth'
    nn_model_name = 'localizer_model.h5'
    
    logic_network_time_per_image = []
    logic_network_accuracy_per_image = []
    ground_truth_of_each_image = []
    neural_network_time_per_image = []
    neural_network_accuracy_per_image = []
    
    logic_network_model = LocalizerLogicCNN()
    logic_network_model.load_state_dict(torch.load(lgn_model_name))
    logic_network_model.to(device)
    logic_network_model.eval()
    
    model = tf.keras.models.load_model(nn_model_name, compile=False)
    
    device = torch.device(device)
    
    for i in range(NUMBER_OF_IMAGES):
        print(f'Image {i}')
        
        lgn_X, gt, p = load_image_and_gt_lgn(dataset_dir, i)
        nn_X, gt, p = load_image_and_gt_nn(dataset_dir, i)
        
        print('Loaded image:', p)
        print('Ground truth bbox (x,y,h,w pixels):', gt)
        ground_truth_of_each_image.append(gt)

        with torch.no_grad():
            lgn_start_time = time.time()
            lgn_pred = logic_network_model(lgn_X.to(device))
            lgn_end_time = time.time()
            
            nn_start_time = time.time()
            nn_pred = model.predict(nn_X)
            nn_end_time = time.time()

        gt_box = np.array(gt)
        
        lgn_pred_box = lgn_pred[0].cpu().numpy()
        
        lgn_pred_box = lgn_pred_box * IMG_DIM
        
        nn_pred_box = np.array(nn_pred[0])
        
        nn_pred_box = nn_pred_box * IMG_DIM
                
        lgn_iou = compute_iou(lgn_pred_box, gt_box)
        
        nn_iou = compute_iou(nn_pred_box, gt_box)
        
        lgn_time_taken = lgn_end_time - lgn_start_time
        
        nn_time_taken = nn_end_time - nn_start_time
        
        logic_network_time_per_image.append(lgn_time_taken)
        
        logic_network_accuracy_per_image.append(lgn_iou)
        
        neural_network_time_per_image.append(nn_time_taken)
        
        neural_network_accuracy_per_image.append(nn_iou)
        
        
    performance = {
        'Image ID': [i for i in range(NUMBER_OF_IMAGES)],
        'Ground Truth Bounding Box': ground_truth_of_each_image,
        'Logic NN Time': logic_network_time_per_image,
        'Logic NN Accuracy': logic_network_accuracy_per_image,
        'Simple NN Time': neural_network_time_per_image,
        'Simple NN Accuracy': neural_network_accuracy_per_image
    }
    
    df = pd.DataFrame(performance)
    
    df.to_excel('Performance Comparison.xlsx')
    
    
compare_performance()