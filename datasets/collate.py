import torch

def multimodal_collate(batch):
    """
    batch is a list of tuples: [(clinical, mri, wsi, label), ...]
    Each element can be a Tensor OR None
    """
    clinical = [item[0] for item in batch]
    mri      = [item[1] for item in batch]
    wsi      = [item[2] for item in batch]
    labels   = torch.stack([item[3] for item in batch])

    # Stack if available, else keep None
    clinical = None if all(v is None for v in clinical) else torch.stack([v for v in clinical if v is not None])
    mri      = None if all(v is None for v in mri) else torch.stack([v for v in mri if v is not None])
    wsi      = None if all(v is None for v in wsi) else torch.stack([v for v in wsi if v is not None])

    return (clinical, mri, wsi), labels

def simple_collate(batch):
    return batch[0]

