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

parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument('--image_folder', type=str, help='경로를 포함한 이미지 폴더')
# parser.add_argument('--start_frame', type=int, help='시작 프레임 번호')
# parser.add_argument('--end_frame', type=int, help='끝 프레임 번호')
# parser.add_argument('--seq', type=str, help='seq 이름')
parser.add_argument('--config', type=str, help='config path')
args = parser.parse_args()

image_folder = args.image_folder
# start_frame = args.start_frame
# end_frame = args.end_frame
# seq = args.seq
load_config = args.config



# image_folder = '0006_1209'
# start_frame = 1211
# end_frame = 1311
# directory_path = f'/home/nas4_user/sungwonhwang/ws_student/taewoong/mars/render_output/novel_view_up/{image_folder}/*'
directory_path = f'/home/nas4_user/minjungkim/Others/kang/KITTI_NSG/example_weights/{image_folder}_render_kitti_tracking_0006_render/renderonly_path_000000/*'

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

# 이미지 경로
pattern = "Rx_0_Rz_0_tz_0"
matching_files = [f for f in glob.glob(directory_path) if pattern in f]
print("matching files : ", len(matching_files))

# print("predict files : ", matching_files)
### GT 이미지
from torch import nn
from PIL import Image
import math
from jaxtyping import Float
from torch import Tensor
from kitti360scripts.helpers.project import CameraPerspective as KITTICameraPerspective
from typing import NamedTuple
from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
img_name = []
load_config = Path(load_config)
_, pipeline, _, _ = eval_setup(
    load_config,
    eval_num_rays_per_chunk=None,
    test_mode="inference",
)
image_file_names = pipeline.datamanager.train_dataset.image_filenames
data_path = pipeline.datamanager.dataparser.data
data_path = str(data_path)
selected_frames = pipeline.datamanager.dataparser.selected_frames
start_frame = selected_frames[0]
end_frame = selected_frames[1]
seq = data_path[-4:]
print("image_folder : ", image_folder)
print("start_frame : " , start_frame)
print("end_frame : ", end_frame)
print("seq : ", seq)

for i in range(len(image_file_names)//2):
    if i % 4 == 0:
        img_name.append(image_file_names[i])

# for i, cam in enumerate(cam_infos):
#     if i < len(cam_infos)//2:
#         if i % 4 == 0:
#             img_name.append(cam.image_name)
            
print("image_name :", len(img_name))
# print("gt_files : ",  img_name)

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

dataset = ImageDataset(img_name, matching_files, transform=transform)
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
        "sequence" : seq,
        "folder" : image_folder,
        "start_frame" : start_frame,
        "end_frame" : end_frame,
        "results" : {
            "average_psnr": np.float32(avg_psnr),
            "average_ssim": np.float32(avg_ssim),
            "average_lpips": np.float32(avg_lpips),
        }
    }
    # "details": [{"gt": os.path.basename(gt_file), "generated": os.path.basename(generated_file), "psnr": p, "ssim": s, "lpips": l} for (gt_file, generated_file), p, s, l in zip(img_name, matching_files, psnr_values, ssim_values, lpips_values)]
}


results_file_path = 'results_up.json'

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