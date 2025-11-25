import torch
import torchio
import torchvision

import random
import numpy as np
import math


class TransformMRI:
    def __init__(self,
                 apply_query_mask=True,
                 modality_keys=['t2w'],
                 **kwargs):
        
        self.N = len(modality_keys)
        self.include = modality_keys
        self.do_mask = apply_query_mask
        self.img_size = np.array(kwargs.get("input_size", [25, 128, 120]))
        
        self.num_ctrl_points = 7
        self.max_disp = (self.img_size / (self.num_ctrl_points - 1))*0.1
        
        self.transforms = torchio.Compose([
            torchio.RescaleIntensity((0,1), include=self.include),
            torchio.RandomFlip(axes=(1,2), flip_probability=kwargs.get("flip_prob", 0.5), include=self.include),
            torchio.RandomAffineElasticDeformation(affine_first=True, 
                                                   affine_kwargs={'scales': (0, 0.1, 0.1), 'degrees': (0,10,10)},
                                                   elastic_kwargs={'num_control_points': self.num_ctrl_points, 
                                                                   'locked_borders': 2,
                                                                   'max_displacement': self.max_disp},
                                                   include=self.include),
            torchio.RandomBiasField(coefficients=kwargs.get("bias_field_coeffs", 1), include=self.include),
            torchio.RandomGhosting(num_ghosts=kwargs.get("num_ghosts", 2), axes=1, intensity=0.1, include=self.include)
        ])

    def _modality_mask(self, query):
        """ Random masking of one modality """
        drop_idx = random.randint(0, self.N-1)
        masked_query = query
        if isinstance(query, dict):
            drop_key = list(query)[drop_idx]
            masked_query[drop_key] = torch.zeros_like(query[drop_key])
        else:
            masked_query[drop_idx] = torch.zeros_like(query[drop_idx])
        return masked_query
    
    def apply_transformation(self, x):
        x_hat = self.transforms(x)
        return x_hat


    def __call__(self, x):
        q = self.apply_transformation(x)
        k = self.apply_transformation(x)

        if self.do_mask:
            q = self._modality_mask(q)

        return q, k


class FroFA:
    def __init__(self, channel_wise=True):
        self.cfrofa = channel_wise
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomHorizontalFlip(p=0.5)
        ])

    def apply_transformation(self, x):
        # Apply transform, can do more stuff here
        if self.cfrofa:
            x_hat = x
            for c in range(x.shape[-1]):
                x_c = x_hat[:,:,:,c]
                x_c = self.transform(x_c)
                x_hat[:,:,:,c] = x_c
        else:
            x_hat = x.permute(0, 3, 1, 2)
            x_hat = self.transform(x_hat)
            x_hat = x_hat.permute(0, 2, 3, 1)

        return x_hat

    def __call__(self, f):
        if f.ndim == 2:
            f = f.unsqueeze(0) # add batch dim 
        B, N, C = f.shape

        # 1. Reshape into 3D format
        H = int(math.floor(math.sqrt(N)))
        N_hat = H*H
        f_hat = f[:, :N_hat, :]
        f_hat = f_hat.view(B, H, H, C)

        # 2. Map to image space
        f_min = f_hat.min()
        f_max = f_hat.max()

        x_f = f_hat - f_min / (f_max - f_min)

        # 3. Apply transformation
        x_hat = self.apply_transformation(x_f)

        # 4. Map back to feature space
        f_hat = x_hat * (f_max - f_min) + f_min

        # 5. Return to original shape
        f_hat = f_hat.view(B, N_hat, C)
        f[:, :N_hat, :] = f_hat

        return f



        

