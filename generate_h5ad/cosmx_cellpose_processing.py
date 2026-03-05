import imageio
import numpy as np
import os
import torch

from pathlib import Path
from tqdm import tqdm
from cellpose import denoise

def normalise(image):
    return (image - image.min()) / (image.max() - image.min())

#---
# Read in images
#---

synthetic_crop_path = Path('/nemo/lab/tedescos/home/shared/Aude/myotube_segmentation_v2/raw_data/dapi_1024')
SAVEPATH = Path('/nemo/lab/tedescos/home/shared/Aude/myotube_segmentation_v2/raw_data/dapi_1024_denoised_deblurred')
SAVEPATH.mkdir(parents=True, exist_ok=True)

image_path_list = sorted(list(synthetic_crop_path.glob('*.png')))
print(f"Found {len(image_path_list)} images")

#---
# Select device
#---
use_gpu = torch.cuda.is_available()
if use_gpu:
    gpu_index = int(os.environ.get("CUDA_DEVICE", "0"))
    torch.cuda.set_device(gpu_index)
    current = torch.cuda.current_device()
    print(f"Using GPU index {current}: {torch.cuda.get_device_name(current)}")
else:
    print("CUDA not available; using CPU")

#---
# Load models
#---
deblur = denoise.DenoiseModel(model_type="deblur_nuclei", gpu=use_gpu)
denoise = denoise.DenoiseModel(model_type="denoise_nuclei", gpu=use_gpu)

BATCH_SIZE = 64

# Process images in batches
for start_idx in tqdm(range(0, len(image_path_list), BATCH_SIZE), desc="Processing batches"):
    batch_paths = image_path_list[start_idx:start_idx + BATCH_SIZE]
    batch_images = [np.array(imageio.imread(p)) for p in batch_paths]

    batch_deblur = deblur.eval(batch_images, channels=None, diameter=40., batch_size=BATCH_SIZE)
    batch_denoise = denoise.eval(batch_deblur, channels=None, diameter=40., batch_size=BATCH_SIZE)

    for img, path in zip(batch_denoise, batch_paths):
        processed_image = normalise(img)
        processed_image = (processed_image * 255).astype(np.uint8).squeeze()
        
        imageio.imsave(SAVEPATH / f"{path.stem}.png", processed_image)