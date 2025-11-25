import torch
import torch.nn as nn
import os
import timm

from torchsummary import summary


class ViTModel(nn.Module):
    def __init__(self, 
                 num_features=196,
                 feature_size=1024, 
                 patch_size=16, 
                 num_modalities=3, 
                 embed_dim=1024, 
                 depth=6, 
                 num_heads=16):
        super(ViTModel, self).__init__()

        self.N = num_features
        
        #self.num_inputs = int((num_features) ** 0.5)**2
        #self.num_dropped = num_features - self.num_inputs

        self.img_size = int((num_features) ** 0.5) * patch_size

        self.base_vit = timm.models.vision_transformer.VisionTransformer(
            img_size=self.img_size,
            patch_size=patch_size,
            in_chans=1,
            num_classes=0,      # no classification head
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=torch.nn.LayerNorm
        )

        self.base_vit.patch_embed = nn.Sequential(
            nn.Linear(feature_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.base_vit.pos_embed = nn.Parameter(
            torch.zeros(1, self.N + 1, embed_dim)
        )

        self.modality_embeddings = nn.Parameter(torch.randn(num_modalities, embed_dim))
        nn.init.trunc_normal_(self.modality_embeddings, std=0.02)


    def forward(self, x, modality_idxs):
        B, N, F = x.shape  # B, num_patches, feature_size

        # Add modality embeddings
        if modality_idxs.ndim == 1:
            modality_emb = self.modality_embeddings[modality_idxs]  # [N, D]
            x = x + modality_emb.unsqueeze(0)  # broadcast across batch
        else:
            modality_emb = self.modality_embeddings[modality_idxs]  # [B, N, D]
            x = x + modality_emb

        #cls_tokens = self.base_vit.cls_token.expand(B, -1, -1)  # [B, 1, D]
        #x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]

        return self.base_vit(x)
    
if __name__ == "__main__":
    model = ViTModel(num_features=2048, feature_size=1024)

    summary(model, [(2048, 1024), (2048,)], dtypes=[torch.float32, torch.long ])

    x = torch.randn(2, 2048, 1024)
    modality_idxs = torch.tensor([0]*1296 + [1]*512 + [2]*240).unsqueeze(0).repeat(2,1)  # Example modality indices

    out = model(x, modality_idxs)
    print(out.shape)  # Expected output shape: [2, 1024]
