import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
import torch
import json
import os
import argparse
import cv2

from data_loader.load_kitti import *

parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument('--prediction_folder', type=str, help='경로를 포함한 이미지 폴더')
parser.add_argument('--gt_folder', type=str, help='gt')
parser.add_argument('--start', type=int, help='first')
parser.add_argument('--last', type=int, help='last')
args = parser.parse_args()

basedir = args.gt_folder
selected_frames = [[args.start],[args.last]]
gt_images, poses, render_poses, [H, W, focal], i_split, visible_objects, objects_meta, render_objects, bboxes,\
      kitti_obj_metadata, time_stamp, render_time_stamp\
        = load_kitti_data(basedir, selected_frames=selected_frames, use_obj=True, row_id=False, remove=-1, use_time=False, exp=False)

prediction_folder = args.prediction_folder


def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    print("image_shape : ", image.shape)
    return image

def calculate_psnr(image1, image2):
    psnr = compare_psnr(image1, image2)
    return psnr

def calculate_ssim(image1, image2):
    data_range = image2.max() - image2.min()
    ssim , _ = compare_ssim(image1, image2, data_range = data_range, channel_axis=0, full=True)# multichannel=True)
    return ssim

lpips_model = lpips.LPIPS(net='vgg').cuda() 

def calculate_lpips(image1, image2):
    image1 = lpips.im2tensor(lpips.load_image(image1))  # 이미지 로드 및 전처리
    image2 = lpips.im2tensor(lpips.load_image(image2))
    if torch.cuda.is_available():
        image1, image2 = image1.cuda(), image2.cuda()  # GPU 사용 설정
    lpips_distance = lpips_model.forward(image1, image2)
    return lpips_distance.item()

from torch import nn
from PIL import Image
import math
from torch import Tensor
from kitti360scripts.helpers.project import CameraPerspective as KITTICameraPerspective
from typing import NamedTuple
from pathlib import Path


## load image to gpu
class ImageDataset(Dataset):
    def __init__(self, gt_files, generated_files, transform=None):
        self.gt_files = gt_files
        self.generated_files = generated_files
        self.transform = transform

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        gt_image = Image.open(self.gt_files[idx]).convert('RGB')
        generated_image = Image.open(self.generated_files[idx]).convert('RGB')

        if self.transform:
            gt_image = self.transform(gt_image)
            generated_image = self.transform(generated_image)

        return gt_image, generated_image

transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# generated_images = [[rgb]]
folders = f'/home/nas4_user/minjungkim/Others/kang/KITTI_NSG/example_weights/{args.prediction_folder}_render_kitti_tracking_0006_render/renderonly_path_000000'
folders = '/home/nas4_user/minjungkim/Others/kang/KITTI_NSG/example_weights/Test_view_render_kitti_tracking_0006_render/renderonly_path_000000'
generated_images = []
img_pths = []
for i in list(os.listdir(folders)):
    image_path = os.path.join(folders,i)
    img_pths.append(image_path)
sorted(img_pths)

generated_images = (np.maximum(np.minimum(np.array(generated_images), 255), 0) / 255.).astype(np.float32)



class ImageDataset(Dataset):
    def __init__(self, gt_files, generated_files, transform=None):
        self.gt_files = gt_files
        self.generated_files = generated_files
        self.transform = transform

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        gt_image = Image.open(self.gt_files[idx]).convert('RGB')
        generated_image = Image.open(self.generated_files[idx]).convert('RGB')

        if self.transform:
            gt_image = self.transform(gt_image)
            generated_image = self.transform(generated_image)

        return gt_image, generated_image

transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
gt_path = '/home/nas4_dataset/3D/KITTI_tracking/training/image_02/0006'
matching_files = sorted(list(os.listdir(gt_path)))
matching_files = matching_files[matching_files.index('000065.png'):matching_files.index('000120.png')+1]
matching_files = [os.path.join(gt_path,num) for num in matching_files]
dataset = ImageDataset(img_pths, matching_files, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False) 


# PSNR, SSIM, LPIPS 계산
psnr_values = []
ssim_values = []
lpips_values = []

for gt_images, generated_images in dataloader:

    gt_images = gt_images.cuda()
    generated_images = generated_images.cuda()

    lpips_distances = lpips_model(gt_images, generated_images)
    lpips_values.extend(lpips_distances.cpu().detach().numpy())

    gt_images = gt_images.cpu().detach().numpy().squeeze()
    generated_images = generated_images.cpu().detach().numpy().squeeze()

    
    psnr_values.append(calculate_psnr(gt_images, generated_images))
    # print(np.asarray(gt_images.shape))
    # print(np.asarray(generated_images.shape))
    ssim_values.append(calculate_ssim(gt_images, generated_images))

avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)
avg_lpips = np.mean(lpips_values)

print(f"Average PSNR: {avg_psnr}")
print(f"Average SSIM: {avg_ssim}")
print(f"Average LPIPS: {avg_lpips}")

new_results = {
    "image_folder ": {
        "sequence" : '0006',
        "start_frame" : args.start,
        "end_frame" : args.last,
        "results" : {
            "average_psnr": np.float32(avg_psnr),
            "average_ssim": np.float32(avg_ssim),
            "average_lpips": np.float32(avg_lpips),
        }
    }
    # "details": [{"gt": os.path.basename(gt_file), "generated": os.path.basename(generated_file), "psnr": p, "ssim": s, "lpips": l} for (gt_file, generated_file), p, s, l in zip(img_name, matching_files, psnr_values, ssim_values, lpips_values)]
}


results_file_path = './results_up.json'

if os.path.exists(results_file_path):
    with open(results_file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:  # 파일이 비어있는 경우
            data = []
else:
    data = []

data.append(new_results)

def convert(o):
    if isinstance(o, np.float32):
        return float(o)
    raise TypeError

with open(results_file_path, 'w') as file:
    json.dump(data, file, indent=4, default = convert)

print('all processes are done')