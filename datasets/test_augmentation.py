from transformation import TransformMRI
import os
import SimpleITK
import glob
import numpy as np
import torchio
import cv2
import torch

import matplotlib.pyplot as plt
import torchvision.utils as vutils

def crop_and_pad(mimg):
    while mimg.ndim < 4:
        mimg = mimg.unsqueeze(0)

    crop_pad = torchio.CropOrPad((20, 128, 120))
    img = crop_pad(mimg)

    return img

def get_mri_vols(radiology_dir, pid, mask=True):
    # Loads all three MRI volumes for a given patient ID
    # Args:
    #   pid: patient ID
    #   mask: Boolean, mask t2w to make it the same size as the others
    # Returns:
    #   dict with keys 'adc', 'hbv', 't2w' and corresponding MRI volumes as torch tensors
    folder = os.path.join(radiology_dir, str(pid))

    # If folder does not exist, return no mri data? For now raise error
    if not os.path.exists(folder):
        raise ValueError(f"Radiology path: {folder} does not exist")    
        #return None

    mri_vols = os.listdir(folder)
    if len(mri_vols) < 4:
        raise ValueError(f"Not enough MRI volumes for patient {pid}")

    x_adc = SimpleITK.ReadImage(glob.glob(os.path.join(folder, "*_adc.mha"))[0])
    x_adc = SimpleITK.GetArrayFromImage(x_adc).astype(np.float32)

    x_hbv = SimpleITK.ReadImage(glob.glob(os.path.join(folder, "*_hbv.mha"))[0])
    x_hbv = SimpleITK.GetArrayFromImage(x_hbv).astype(np.float32)   

    x_t2w = SimpleITK.ReadImage(glob.glob(os.path.join(folder, "*_t2w.mha"))[0])
    x_t2w = SimpleITK.GetArrayFromImage(x_t2w).astype(np.float32)

    if mask:
        x_mask = SimpleITK.ReadImage(glob.glob(os.path.join(folder, "*_mask.mha"))[0])
        x_mask = SimpleITK.GetArrayFromImage(x_mask)

        idxs = np.where(x_mask == 1)
    
        ymid = int(np.median(np.sort(idxs[1])))
        xmid = int(np.median(np.sort(idxs[2])))

        x_t2w = x_t2w[:, ymid-64:ymid+64, xmid-60:xmid+60]

    x_adc = crop_and_pad(torch.tensor(x_adc))
    x_hbv = crop_and_pad(torch.tensor(x_hbv))
    x_t2w = crop_and_pad(torch.tensor(x_t2w))

    return {
        'adc': x_adc,
        'hbv': x_hbv,
        't2w': x_t2w
    }


def old():
    radiology_dir = "/Volumes/PA_CPGARCHIVE/projects/chimera/_aws/task1/radiology/images"
    
    transform = TransformMRI(apply_query_mask=True, modality_keys=['adc', 'hbv', 't2w'], 
                             num_ghosts=2, flip_probability=0.5)
    
    for pid in [1003, 1010, 1011, 1021]:
        mri_data = get_mri_vols(radiology_dir, pid)
        query, key = transform(mri_data)

        s = 12
        t2w_img = np.array(mri_data['t2w'].squeeze()[s,:,:])
        t2w_img = cv2.normalize(t2w_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("t2w image", t2w_img)

        adc_img = np.array(mri_data['adc'].squeeze()[s,:,:])
        adc_img = cv2.normalize(adc_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("adc image", adc_img)

        hbv_img = np.array(mri_data['hbv'].squeeze()[s,:,:])
        hbv_img = cv2.normalize(hbv_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("hbv image", hbv_img)


        t2w_img = np.array(query['t2w'].squeeze()[s,:,:])
        t2w_img = cv2.normalize(t2w_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("t2w query", t2w_img)

        adc_img = np.array(query['adc'].squeeze()[s,:,:])
        adc_img = cv2.normalize(adc_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("adc query", adc_img)

        hbv_img = np.array(query['hbv'].squeeze()[s,:,:])
        hbv_img = cv2.normalize(hbv_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("hbv query", hbv_img)

        cv2.waitKey(0)  # Wait for a key press to show the next
        cv2.destroyAllWindows()


if __name__ == "__main__":
    radiology_dir = "/Volumes/PA_CPGARCHIVE/projects/chimera/_aws/task1/radiology/images"
    
    transform = TransformMRI(
        apply_query_mask=True, 
        modality_keys=['adc', 'hbv', 't2w'], 
        num_ghosts=2, 
        flip_probability=0.5
    )
    
    for pid in [1003, 1010, 1011, 1021]:
        mri_data = get_mri_vols(radiology_dir, pid)
        query, key = transform(mri_data)

        s = 12
        
        # ---- Collect slices ----
        imgs = []

        # original
        for mod in ['t2w', 'adc', 'hbv']:
            img = mri_data[mod].squeeze()[s, :, :]
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            imgs.append(torch.tensor(img).unsqueeze(0))  # shape: (1, H, W)

        # transformed
        for mod in ['t2w', 'adc', 'hbv']:
            img = query[mod].squeeze()[s, :, :]
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            imgs.append(torch.tensor(img).unsqueeze(0))  # shape: (1, H, W)

        # ---- Stack and make grid ----
        batch = torch.stack(imgs)  # shape: (6, 1, H, W)

        grid = vutils.make_grid(batch, nrow=3, padding=5)  # 3 images per row

        # ---- Show using matplotlib ----
        plt.figure(figsize=(10, 6))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
        plt.axis("off")
        plt.title(f"PID {pid} | Original (top row) vs Query (bottom row)")
        plt.show()
