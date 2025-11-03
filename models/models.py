import torch
import torch.nn as nn
import os

from perceiver_pytorch import Perceiver, PerceiverIO

import torchsummary

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
            x = x.unsqueeze(0)
        return x
    

class ProjectionMRI2d(nn.Module):
    def __init__(self,
                 input_size, 
                 input_channels,
                 hidden_dim, 
                 output_dim):
        super(ProjectionMRI2d, self).__init__()

        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear((input_size[0] // 2) * (input_size[1] // 2) * hidden_dim, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
    
class ProjectionMRI3d(nn.Module):
    def __init__(self,
                 input_size, 
                 input_channels,
                 hidden_dim, 
                 output_dim):
        super(ProjectionMRI3d, self).__init__()

        if len(input_size) != 3:
            raise ValueError("input_size must be a tuple of three dimensions for 3D data.")
        
        self.zdim, self.ydim, self.xdim = input_size

        self.conv1 = nn.Conv3d(input_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(hidden_dim)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear((input_size[0] // 2) * (input_size[1] // 2) * (input_size[2] // 2) * hidden_dim, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x
    


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

        torchsummary.summary(self.perceiver, (5500, perceiver_input_dim))

    def forward(self, clinical_data, mri_data, wsi_data):
        clinical_features = self.clinical_mlp(clinical_data)
        #mri_features = self.mri_projection(mri_data)

        # Combine features (e.g., concatenate)
        combined_features = torch.cat((wsi_data, clinical_features, mri_data), dim=1)

        # Pass through Perceiver
        if combined_features.dim() == 2:
            combined_features = combined_features.unsqueeze(0)
        output = self.perceiver(combined_features)  # Add sequence dimension

        return output


class MMCLSModel(MMCLRModel):
    def __init__(self,
                 threshold=0.5,
                  *args, **kwargs):
        super(MMCLSModel, self).__init__(*args, **kwargs)
        # Additional layers or modifications for classification can be added here

        input_dim = kwargs.get('perceiver_output_dim', 1000)

        self.classifier = nn.Linear(input_dim, 1) # Example: binary classification
        self.threshold = threshold


    def forward(self, input_data):

        assert isinstance(input_data, tuple) and len(input_data) == 3, "Input must be a tuple of (clinical_data, mri_data, wsi_data)"

        clinical_data, mri_data, wsi_data = input_data
        x = super(MMCLSModel, self).forward(clinical_data, mri_data, wsi_data)
        logits = self.classifier(x)

        y_prob = torch.sigmoid(logits)
        y_hat = torch.where(y_prob > self.threshold, 1, 0)

        return logits, y_prob, y_hat



def test_mri_projection():
    # Try with full 3D projection of each MRI volume
    model = ProjectionMRI3d(input_size=(19, 128, 120), input_channels=1, hidden_dim=128, output_dim=1024)
    dummy_input = torch.randn(3, 1, 19, 128, 120)

    torchsummary.summary(model, (1, 19, 128, 120))

    output = model(dummy_input)
    print(output.shape)  # should be [3, 1024]

#test_mri_projection()

def test_perceiverio():
    model = PerceiverIO(
        dim = 1024,                    # dimension of sequence to be encoded
        queries_dim = 1024,            # dimension of decoder queries
        logits_dim = 1000,            # dimension of final logits
        depth = 6,                   # depth of net
        num_latents = 1,           # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim = 512,            # latent dimension
        cross_heads = 1,             # number of heads for cross attention. paper said 1
        latent_heads = 8,            # number of heads for latent self attention, 8
        cross_dim_head = 64,         # number of dimensions per cross attention head
        latent_dim_head = 64,        # number of dimensions per latent self attention head
        weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
        seq_dropout_prob = 0.2       # fraction of the tokens from the input sequence to dropout (structured dropout, for saving compute and regularizing effects)
    )

    torchsummary.summary(model, (5500, 1024))

    dummy_input = torch.randn(1, 5546, 1024)  # batch size 1, 5546 channels, sequence length 1024
    output = model(dummy_input)
    print(output.shape)  # should be [1, 1, 512]


def test_perceiver():

    from perceiver_pytorch import Perceiver

    model = Perceiver(
        input_channels = 1024,          # number of channels for each token of the input
        input_axis = 1,              # number of axis for input data (2 for images, 3 for video)
        num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
        max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
        depth = 4,                   # depth of net. The shape of the final attention mechanism will be:
                                    #   depth * (cross attention -> self_per_cross_attn * self attention)
        num_latents = 1,           # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim = 1024,            # latent dimension
        cross_heads = 1,             # number of heads for cross attention. paper said 1
        latent_heads = 6,            # number of heads for latent self attention, 8
        cross_dim_head = 32,         # number of dimensions per cross attention head
        latent_dim_head = 32,        # number of dimensions per latent self attention head
        num_classes = 1000,          # output number of classes
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
        fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
        self_per_cross_attn = 2      # number of self attention blocks per cross attention
    )

    torchsummary.summary(model, (10, 1024))

    img = torch.randn(1, 5500, 1024) # 1 imagenet image, pixelized

    output = model(img) # (1, 1000)

    print(output.shape)




def test_mmclr_model():
    print("Testing MMCLR Model")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    #    print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = MMCLRModel(
        clinical_input_dim=18,
        clinical_hidden_dim=1024,
        clinical_output_dim=1024,
        device=device
    ).to(device)

    clinical_data = torch.randn(1, 18).to(device)  # batch size 1, clinical features 18
    mri_data = torch.randn(1, 1, 1024).to(device)  # batch size 3, 1 MRI image with 19 slices, 128x120
    wsi_data = torch.randn(1, 5450, 1024).to(device)  # batch size 1, 5450 WSI features 1024

    output = model(clinical_data, mri_data, wsi_data)
    print(output.shape)  # should be [1, 1, 1000]


def test_mmcls_model():
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

    model = MMCLSModel(
        clinical_input_dim=18,
        clinical_hidden_dim=1024,
        clinical_output_dim=1024,
        device=device
    ).to(device)

    clinical_data = torch.randn(1, 18).to(device)  # batch size 1, clinical features 18
    mri_data = torch.randn(1, 1, 1024).to(device)  # batch size 3, 1 MRI image with 19 slices, 128x120
    wsi_data = torch.randn(1, 5450, 1024).to(device)  # batch size 1, 5450 WSI features 1024
    data = (clinical_data, mri_data, wsi_data)

    logits, y_prob, y_hat = model(data)
    print(y_prob)  


if __name__ == "__main__":
    test_mmcls_model()