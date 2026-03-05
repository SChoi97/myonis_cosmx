import imageio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

datapath = Path('/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/cosmx_t5r5/Laminopathy_CosMx_T5R5_11_08_2025_11_08_44_887/DecodedFiles/SP24153_T5R5_240725/20250724_142445_S1/CellStatsDir/Morphology2D')
savepath = Path('/nemo/lab/tedescos/home/users/chois1/nanostring/cosmx/cosmx_6k_2025/processed_files/t5r5/processed_images')

MYHC_PATH = savepath / 'myhc'
MYHC_PATH.mkdir(exist_ok=True, parents=True)
DAPI_PATH = savepath / 'dapi'
DAPI_PATH.mkdir(exist_ok=True, parents=True)

myhc_rescale_max = 40000
dapi_rescale_max = 5000
crop_size = 1024

image_path_list = sorted(list(datapath.glob('*.TIF')))

print(f"Found {len(image_path_list)} images")

for image_path in tqdm(image_path_list):

    #Read in image
    image_name = image_path.stem
    image = np.array(imageio.imread(image_path))
    myhc_image = image[1]
    dapi_image = image[4]
    
    #Rescale the images
    myhc_image = np.clip(myhc_image, 0, myhc_rescale_max)
    dapi_image = np.clip(dapi_image, 0, dapi_rescale_max)
    
    # #16-bit to 8-bit
    # myhc_image = (myhc_image / myhc_rescale_max) * 255
    # dapi_image = (dapi_image / dapi_rescale_max) * 255

    #Generate crops
    height, width = myhc_image.shape
    patch_idx = 0
    for i in range(0, height, crop_size):
        for j in range(0, width, crop_size):
            myhc_crop = myhc_image[i:i+crop_size, j:j+crop_size]
            dapi_crop = dapi_image[i:i+crop_size, j:j+crop_size]

            if myhc_crop.shape == (crop_size, crop_size):
                # Save crops
                myhc_savename = MYHC_PATH / f"{image_name}_patch_{patch_idx}.png"
                dapi_savename = DAPI_PATH / f"{image_name}_patch_{patch_idx}.png"

                # imageio.imwrite(myhc_savename, myhc_crop.astype(np.uint8))
                # imageio.imwrite(dapi_savename, dapi_crop.astype(np.uint8))

                imageio.imwrite(myhc_savename, myhc_crop)
                imageio.imwrite(dapi_savename, dapi_crop)
                
                patch_idx += 1
    



