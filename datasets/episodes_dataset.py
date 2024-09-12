import argparse
import warnings

from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os
import os.path as osp
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class EpisodesDataset(Dataset):
    def __init__(self, root, mode, sample_length=1, res=128, episodic=False):
        assert mode in ['train', 'val', 'valid']
        if mode == 'valid':
            mode = 'val'
        self.root = os.path.join(root, mode)
        self.res = res
        self.episodic = episodic

        self.mode = mode
        self.sample_length = sample_length

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(file)
            except ValueError:
                continue

        self.folders.sort(key=lambda x: int(x))

        self.episode_images = []
        self.episode2offset = [0]
        self.index2episode = []
        for i, f in enumerate(self.folders):
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, '*.png')))
            actual_length = len(paths) - self.sample_length + 1
            if actual_length <= 0:
                warnings.warn(f'Drop episode {dir_name} with length={len(paths)} as it too short for sample_length={self.sample_length}')
                continue

            get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_num)
            self.episode_images.append(paths)
            self.index2episode.extend([len(self.episode_images) - 1] * actual_length)
            self.episode2offset.append(self.episode2offset[-1] + actual_length)

    def __getitem__(self, index):
        imgs = []
        ep = self.index2episode[index]
        if self.mode == 'train' or not self.episodic:
            # Implement continuous indexing
            offset = self.episode2offset[ep]
            begin = index - offset
            end = begin + self.sample_length
            for image_index in range(begin, end):
                img = Image.open(self.episode_images[ep][image_index])
                img = img.resize((self.res, self.res))
                img = transforms.ToTensor()(img)[:3]
                imgs.append(img)
        else:
            for path in self.episode_images[ep]:
                img = Image.open(path)
                img = img.resize((self.res, self.res))
                img = transforms.ToTensor()(img)[:3]
                imgs.append(img)

        img = torch.stack(imgs, dim=0).float()
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, pos, size, id, in_camera

    def __len__(self):
        if self.mode == 'train' or not self.episodic:
            # Number of available sequences of length self.sample_length
            return len(self.index2episode)
        else:
            # Number of episodes
            return len(self.episode_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--mode', choices=['train', 'val'], default='train')
    parser.add_argument('--sample_length', type=int, default=1)

    args = parser.parse_args()
    root = args.path
    mode = args.mode
    sample_length = args.sample_length
    ds = EpisodesDataset(root, mode, sample_length=sample_length, res=64)
    print('Length:', len(ds))
    img, pos, size, id, in_camera = ds[0]
    print('Shape:', img.size(), 'Mean:', img.mean())
