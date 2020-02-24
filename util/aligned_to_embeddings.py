import torch
import torchvision.transforms.functional as F
import argparse
from facenet_pytorch import InceptionResnetV1
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from face_recognition import preprocessing
from util import save_embeddings_data


class ToTensor(object):

    def __call__(self, img):
        return (F.to_tensor(img) * 255).type(torch.uint8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', required=True,
                        help='Root folder where aligned images are. This folder contains sub-folders for each class.')
    parser.add_argument('--output-folder', required=True, help='Output folder where aligned images will be saved.')
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    trans = transforms.Compose([
        ToTensor(),
        preprocessing.FixedImageStandardization()
    ])

    dataset = datasets.ImageFolder(root=args.input_folder, transform=trans)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    data_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=False, num_workers=1)

    embeddings = None
    for imgs, labels in data_loader:
        batch_embeddings = resnet(imgs)
        embeddings = batch_embeddings if embeddings is None else torch.cat([embeddings, batch_embeddings])

    paths = []
    labels = []
    for path, label in dataset.imgs:
        paths.append(path)
        labels.append(idx_to_class[label])

    save_embeddings_data(args.output_folder, embeddings, labels, paths, dataset.class_to_idx)


if __name__ == '__main__':
    main()
