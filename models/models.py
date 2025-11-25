import torch
import torch.nn as nn
import os
import timm

from perceiver_pytorch import Perceiver, PerceiverIO

from .vit import ViTModel

from .moco import MoCo

from torchinfo import summary
from monai.networks.nets import resnet
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

# simple MLP for clinical data features
class CMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(CMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x
    

class MRIEncoder(nn.Module):
    def __init__(self,
                 multiparametric=True,
                 path_to_weights=None
                 ):
        super(MRIEncoder, self).__init__()


        self.mp=multiparametric
        if path_to_weights is not None:
            self.resnet = resnet.resnet18(spatial_dims=3, n_input_channels=1, feed_forward=False)
            state_dict = torch.load(path_to_weights, weights_only=True)
            consume_prefix_in_state_dict_if_present(state_dict=state_dict, prefix="net.")
            self.resnet.load_state_dict(state_dict, strict=False)
        else:
            self.resnet = resnet.resnet18(pretrained=True, spatial_dims=3, n_input_channels=1, feed_forward=False,
                                shortcut_type='A')
        for p in self.resnet.parameters():
            p.requires_grad = False
        self.resnet.eval()

    def forward(self, sample):
        with torch.no_grad():
            if self.mp:
                emb_adc = self.resnet(sample['adc']).unsqueeze(1)
                emb_hbv = self.resnet(sample['hbv']).unsqueeze(1)
                emb_t2w = self.resnet(sample['t2w']).unsqueeze(1)

                embedding = torch.cat((emb_adc, emb_hbv, emb_t2w), dim=1)
            else:
                embedding = self.resnet(sample['t2w']).unsqueeze(1)

        return embedding
        
    

class MMCLRModel(nn.Module):
    def __init__(self,
                 clinical_input_dim,
                 device,
                 clinical_hidden_dim=512,
                 clinical_output_dim=1024,
                 #mri_input_size=(19, 128, 120),
                 #mri_input_channels=1,
                 perceiver_input_dim=1024,
                 perceiver_output_dim=1000,
                 perceiver_depth=6,
                 perceiver_latent_dim=1024,
                 perceiver_cross_heads=1,
                 perceiver_latent_heads=8,
                 perceiver_attn_drop=0.0):
        super(MMCLRModel, self).__init__()


        self.clinical_mlp = CMLP(clinical_input_dim, clinical_hidden_dim, clinical_output_dim).to(device)
        #self.mri_projection = ProjectionMRI3d(mri_input_size, mri_input_channels, hidden_dim=128, output_dim=1024)
        self.clinical_dim = clinical_input_dim
        
        self.perceiver = Perceiver(
            input_channels = perceiver_input_dim,          # number of channels for each token of the input
            input_axis = 1,              # number of axis for input data (1 for features 2 for images)
            num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
            max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
            depth = perceiver_depth,     # depth of net. The shape of the final attention mechanism will be:
                                        #   depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents = 1,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = perceiver_latent_dim,   # latent dimension
            cross_heads = perceiver_cross_heads,             # number of heads for cross attention. paper said 1
            latent_heads = perceiver_latent_heads,            # number of heads for latent self attention, 8
            cross_dim_head = 32,         # number of dimensions per cross attention head
            latent_dim_head = 32,        # number of dimensions per latent self attention head
            num_classes = perceiver_output_dim,          # output number of classes
            attn_dropout = perceiver_attn_drop,
            ff_dropout = 0.,
            weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn = 2      # number of self attention blocks per cross attention
        ).to(device)

        summary(self.perceiver, (5500, perceiver_input_dim))

    def forward(self, clinical_data, mri_data, wsi_data):
        error = False
        if clinical_data.shape[-1] != self.clinical_dim:
            print(f"Warning: clinical data dimension {clinical_data.shape} does not match expected {self.clinical_dim} \n Replacing with zeros.")
            clinical_data = torch.zeros((clinical_data.shape[0], self.clinical_dim)).to(clinical_data.device)
            print(f"New clinical data shape: {clinical_data.shape}")
            error = True
        clinical_features = self.clinical_mlp(clinical_data)
        #mri_features = self.mri_projection(mri_data)

        if error:
            print(f"Clinical features shape after MLP: {clinical_features.shape}")

        # Combine features (e.g., concatenate)
        if len(mri_data.shape) == 2:
            mri_data = mri_data.unsqueeze(0)
        if len(wsi_data.shape) == 2:
            wsi_data = wsi_data.unsqueeze(0)
        if len(clinical_features.shape) == 2:
            clinical_features = clinical_features.unsqueeze(0)

        if error:
            print(f"Shapes before concatenation: WSI: {wsi_data.shape}, Clinical: {clinical_features.shape}, MRI: {mri_data.shape}")
            
        combined_features = torch.cat((wsi_data, clinical_features, mri_data), dim=1)

        # Pass through Perceiver
        if combined_features.dim() == 2:
            combined_features = combined_features.unsqueeze(0)
        output = self.perceiver(combined_features)  # Add sequence dimension

        return output

class MMViTModel(nn.Module):
    def __init__(self,
                 clinical_input_dim,
                 device,
                 clinical_hidden_dim=512,
                 feature_dim=1024,
                 num_features=2306, # 48x48 grid of patches + clinical + mri
                 num_modalities=3,
                 mri_feature_encoder="unet",
                 *args,
                 **kwargs
                 ):
        super(MMViTModel, self).__init__()

        self.clinical_mlp = CMLP(clinical_input_dim, clinical_hidden_dim, feature_dim)

        self.clinical_dim = clinical_input_dim
        self.N = num_features
        self.F = feature_dim

        self.mri_fe = mri_feature_encoder

        if self.mri_fe == 'resnet':
            self.mri_encoder = MRIEncoder(multiparametric=True, 
                                          path_to_weights=kwargs.get("mri_resnet_weights", None))
            self.projector = nn.Sequential(
                nn.Linear(512, self.F),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            )
        else:
            self.mri_encoder = nn.Identity()
            self.projector = nn.Identity()


        self.vit_model = ViTModel(num_features=num_features, 
                                  feature_size=feature_dim, 
                                  num_modalities=num_modalities,
                                  patch_size=16,
                                  embed_dim=1024,
                                  depth=6,
                                  num_heads=16)
        
    def summarize(self):
        summary(self.vit_model, [(self.N, self.F), (self.N,)], dtypes=[torch.float32, torch.long])


    def forward(self, input_data): #clinical_data, mri_data, wsi_data):
        clinical_data, mri_data, wsi_data = input_data

        # Process clinical features
        error = False
        if clinical_data.shape[-1] != self.clinical_dim:
            print(f"Warning: clinical data dimension {clinical_data.shape} does not match expected {self.clinical_dim} \n Replacing with zeros.")
            clinical_data = torch.zeros((clinical_data.shape[0], self.clinical_dim)).to(clinical_data.device)
            print(f"New clinical data shape: {clinical_data.shape}")
            error = True
        clinical_features = self.clinical_mlp(clinical_data)
        if error:
            print(f"Clinical features shape after MLP: {clinical_features.shape}")

        # Processs MRI features
        with torch.no_grad():
            mri_features = self.mri_encoder(mri_data)
        mri_data = self.projector(mri_features)

        # Combine features (e.g., concatenate)
        if len(mri_data.shape) == 2:
            mri_data = mri_data.unsqueeze(1)
        if len(wsi_data.shape) == 2:
            wsi_data = wsi_data.unsqueeze(1)
        if len(clinical_features.shape) == 2:
            clinical_features = clinical_features.unsqueeze(1)

        if error:
            print(f"Shapes before concatenation: WSI: {wsi_data.shape}, Clinical: {clinical_features.shape}, MRI: {mri_data.shape}")
            
        x = torch.cat((wsi_data, clinical_features, mri_data), dim=1)

        # Create modality indices
        c_dim = clinical_features.shape[1]
        r_dim = mri_data.shape[1]
        p_dim = wsi_data.shape[1]

        assert c_dim + r_dim + p_dim == self.N, f"Sum of modality dimensions ({c_dim}, {r_dim}, {p_dim}) must equal total number of features ({self.N})"

        modality_idxs = torch.tensor([2]*p_dim + [0]*c_dim + [1]*r_dim).unsqueeze(0).repeat(x.shape[0],1).to(x.device)

        # Pass through ViT
        if x.dim() == 2:
            x = x.unsqueeze(0)
        output = self.vit_model(x, modality_idxs)  

        return output

class MMCLSModel(nn.Module):
    def __init__(self,
                 base_model,
                 cls_input_dim,
                 threshold=0.5,
                  *args, **kwargs):
        super(MMCLSModel, self).__init__()
        self.base_model = base_model

        input_dim = cls_input_dim

        self.device = kwargs.get('device', torch.device('cpu'))
        self.classifier = nn.Linear(input_dim, 1).to(self.device) # Example: binary classification
        self.threshold = threshold


    def forward(self, input_data):

        assert isinstance(input_data, (tuple, list)) and len(input_data) == 3, "Input must be a tuple of (clinical_data, mri_data, wsi_data)"

        x = self.base_model(input_data)
        logits = self.classifier(x)

        y_prob = torch.sigmoid(logits)
        y_hat = torch.where(y_prob > self.threshold, 1, 0)

        return logits, y_prob, y_hat


class MoCoWrapper(MoCo):
    def __init__(self, 
                    base_encoder,
                    device,
                    encoder_args: dict = {},
                    dim: int = 1024, # feature dimension
                    K: int = 1000, # queue size
                    m: float = 0.999, # moco momentum
                    T: float = 0.07, # softmax temperature
                    do_batch_shuffle: bool = False, # Do batch shuffling, does not make sense with batch size = 1?
                    ):
        super(MoCoWrapper, self).__init__(base_encoder=base_encoder,
                                          encoder_args=encoder_args,
                                          device=device,
                                        dim=dim,
                                        K=K,
                                        m=m,
                                        T=T)
        
        self.do_batch_shuffle = do_batch_shuffle
        self.device = device
        
    def forward(self, input_q, input_k):
        """
        Input:
            input_q: a batch of query features (Student input, missing data)
            input_k: a batch of key features (Teacher input, complete data)
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(input_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            if self.do_batch_shuffle:
                input_k, idx_unshuffle = self._batch_shuffle_ddp(input_k)

            k = self.encoder_k(input_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if self.do_batch_shuffle:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, q
        

class ClassifierHead(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_output=1,
                 dropout=0.25,
                 op=0.5):
        super().__init__()
            
        if hidden_dim is not None:
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_output)
            )
        else:
            self.fc = nn.Linear(input_dim. num_output)

        self.op = op

    def forward(self, x):
        logits = self.fc(x)
        y_prob = torch.sigmoid(logits)
        y_hat = torch.where(y_prob > self.op, 1, 0)
        return logits, y_prob, y_hat



def test_mm_model(model_type="mmclr"):
    print("Testing MM Model")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    #    print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    if model_type.lower() == "mmclr":
        model = MMCLRModel(
            clinical_input_dim=15,
            clinical_hidden_dim=1024,
            clinical_output_dim=1024,
            device=device
        ).to(device)
    elif model_type.lower() == "vit":
        model = MMViTModel(
            clinical_input_dim=15,
            clinical_hidden_dim=1024,
            feature_dim=1024,
            num_features=2306,
            num_modalities=3,
            device=device
        ).to(device)

    clinical_data = torch.randn(1, 15).to(device)  # batch size 1, clinical features 15
    mri_data = torch.randn(1, 1, 1024).to(device)  # batch size 1, 1 MRI image with 19 slices, 128x120
    wsi_data = torch.randn(1, 48*48, 1024).to(device)  # batch size 1, 2304 WSI features (48x48 grid of patches) 1024

    output = model(clinical_data, mri_data, wsi_data)
    print(output.shape)  


def test_mmcls_model(model_type="vit"):
    print("Testing MMCLS Model")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    #    print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    if model_type.lower() == "mmclr":
        base_model = MMCLRModel(
            clinical_input_dim=15,
            clinical_hidden_dim=1024,
            clinical_output_dim=1024,
            device=device
        ).to(device)
    elif model_type.lower() == "vit":
        base_model = MMViTModel(
            clinical_input_dim=15,
            clinical_hidden_dim=1024,
            feature_dim=1024,
            num_features=2306,
            num_modalities=3,
            device=device
        ).to(device)

    model = MMCLSModel(
        base_model=base_model,
        cls_input_dim=1024,
        threshold=0.5
    ).to(device)

    clinical_data = torch.randn(1, 15).to(device)  # batch size 1, clinical features 15
    mri_data = torch.randn(1, 1, 1024).to(device)  # batch size 1, 1 MRI image with 19 slices, 128x120
    wsi_data = torch.randn(1, 2304, 1024).to(device)  # batch size 1, 2304 WSI features (48x48 grid of patches) 1024
    data = (clinical_data, mri_data, wsi_data)

    logits, y_prob, y_hat = model(data)
    print(y_prob)  


def test_moco():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    base_model = MMViTModel
    
    encoder_args = {
        'clinical_input_dim': 15,
        'clinical_hidden_dim': 1024,
        'num_features': 2306,
        'num_modalities': 3,
        'mri_feature_encoder': 'resnet'
    }

    moco_model = MoCoWrapper(base_encoder=base_model,
                             encoder_args=encoder_args,
                             device=device).to(device)
    

    batch_size = 4
    clinical_data_q = torch.ones(batch_size, 15).to(device)  # batch size 1, clinical features 15
    mri_data_q = torch.zeros(batch_size, 1024).to(device)  # batch size 1, 1 MRI image with 19 slices, 128x120
    wsi_data_q = torch.randn(batch_size, 2304, 1024).to(device)  # batch size 1, 2304 WSI features (48x48 grid of patches) 1024
    input_q = (clinical_data_q, mri_data_q, wsi_data_q) 

    clinical_data_k = torch.ones(batch_size, 15).to(device)  # batch size 1, clinical features 15
    mri_data_k = torch.zeros(batch_size, 1, 1024).to(device)
    wsi_data_k = torch.randn(batch_size, 2304, 1024).to(device)  # batch size 1, 2304 WSI features (48x48 grid of patches) 1024
    input_k = (clinical_data_k, mri_data_k, wsi_data_k)

    logits, labels, query = moco_model(input_q, input_k)
    print(logits.shape)
    print(labels.shape)
    print(query.shape)

    summary(moco_model, input_data=(input_q, input_k))


if __name__ == "__main__":
    #test_mmcls_model(model_type="vit")
    test_moco()