import torch
import matplotlib.pyplot as plt
import torchvision
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x

def data_visualize(val_data):
    # Visualize some examples
    NUM_IMAGES = 4
    CIFAR_images = torch.stack([val_data[idx][0] for idx in range(NUM_IMAGES)], dim=0)
    img_grid = torchvision.utils.make_grid(CIFAR_images, nrow=4, normalize=True, pad_value=0.9)
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.title("Image examples of the CIFAR10 dataset")
    plt.imshow(img_grid)
    plt.axis("off")
    plt.show()
    plt.close()

def patch_data_visualize(patch_data):
    fig, ax = plt.subplots(patch_data.shape[0], 1, figsize=(14, 3))
    fig.suptitle("Images as input sequences of patches")
    for i in range(patch_data.shape[0]):
        img_grid = torchvision.utils.make_grid(patch_data[i], nrow=64, normalize=True, pad_value=0.9)
        img_grid = img_grid.permute(1, 2, 0)
        ax[i].imshow(img_grid)
        ax[i].axis("off")
    plt.show()
    plt.close()

def plot_TSNE(features_s, features_t):
    f_s_cpu = features_s.detach().cpu().numpy()
    f_t_cpu = features_t.detach().cpu().numpy()
    features_tsne_s = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(f_s_cpu)
    features_tsne_t = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(f_t_cpu)
    source_mean = torch.mean(features_s, 0)
    target_mean = torch.mean(features_t, 0)
    mmd_similarity = torch.sum((source_mean - target_mean) ** 2)
    return features_tsne_s, features_tsne_t, mmd_similarity

