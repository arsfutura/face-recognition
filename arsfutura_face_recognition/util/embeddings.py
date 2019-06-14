import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1


def main():
    torch.set_grad_enabled(False)
    aligned_images = datasets.ImageFolder('test', transform=transforms.ToTensor())
    aligned_images.idx_to_class = {v: k for k, v in aligned_images.class_to_idx.items()}
    data_loader = DataLoader(aligned_images, batch_size=32)
    facenet = InceptionResnetV1(pretrained='vggface2').eval()

    embeddings = None
    labels = []
    for x, y in data_loader:
        embeddings = facenet(x) if embeddings is None else torch.cat([embeddings, facenet(x)], dim=0)
        labels += map(lambda idx: aligned_images.idx_to_class[idx], y.detach().numpy().tolist())
    np.savetxt('embeddings.txt', embeddings.detach().numpy())
    np.savetxt('labels.txt', np.array(labels, dtype=np.str).reshape(-1, 1), fmt="%s")


if __name__ == '__main__':
    main()
