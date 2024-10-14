import argparse
import warnings

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os
import os.path as osp
import torch
from PIL import Image, ImageFile

from datasets.dataset_item import DatasetItem

ImageFile.LOAD_TRUNCATED_IMAGES = True


class EpisodesDataset(Dataset):
    def __init__(self, root, mode, sample_length=1, res=128, episodic_on_train=False, episodic_on_val=False,
                 use_actions=False):
        assert mode in ['train', 'val', 'valid']
        if mode == 'valid':
            mode = 'val'
        self.root = os.path.join(root, mode)
        self.res = res
        self.use_actions = use_actions

        self.mode = mode
        self.episodic = (self.mode == 'train' and episodic_on_train) or (self.mode == 'val' and episodic_on_val)
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
        self.actions = []
        action_dim = None
        action_type = None
        min_action = np.inf
        max_action = -np.inf
        for i, f in enumerate(self.folders):
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, '*.png')))
            actual_length = len(paths) - self.sample_length + 1
            if actual_length <= 0:
                warnings.warn(
                    f'Drop episode {dir_name} with length={len(paths)} as it too short for sample_length={self.sample_length}')
                continue

            if self.use_actions:
                actions_path = os.path.join(dir_name, 'actions.npy')
                assert os.path.exists(actions_path), f'{os.path.abspath(actions_path)} does not exists.'
                episode_actions = np.load(actions_path)
                self.actions.append(episode_actions)
                if action_dim is None:
                    action_dim = episode_actions.shape[1:]
                else:
                    assert episode_actions.shape[1:] == action_dim, \
                        f'Action dimension mismatch. Expected: {action_dim}. Actual: {episode_actions.shape[1:]}.'

                if action_type is None:
                    action_type = episode_actions.dtype
                else:
                    assert episode_actions.dtype == action_type, \
                        f'Action type mismatch. Expected: {action_type}. Actual: {episode_actions.dtype}.'

                if np.issubdtype(action_type, np.integer):
                    min_action = min(min_action, episode_actions.min())
                    max_action = max(max_action, episode_actions.max())

            get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_num)
            self.episode_images.append(paths)
            self.index2episode.extend([len(self.episode_images) - 1] * actual_length)
            self.episode2offset.append(self.episode2offset[-1] + actual_length)

        if np.issubdtype(action_type, np.integer):
            assert min_action == 0, \
                f'For discrete action spaces the minimal action is expected to be 0. Actual: {min_action}.'
            self.n_actions = max_action + 1
            self.action_space = 'discrete'
        else:
            self.action_space = 'continuous'

    def __getitem__(self, index):
        if self.episodic:
            ep = index
            begin = 0
            end = len(self.episode_images[ep])
        else:
            ep = self.index2episode[index]
            # Implement continuous indexing
            offset = self.episode2offset[ep]
            begin = index - offset
            end = begin + self.sample_length

        if self.use_actions:
            action = torch.as_tensor(self.actions[ep][begin: end - 1])
            if self.action_space == 'discrete':
                action = torch.nn.functional.one_hot(action, num_classes=self.n_actions)

            action = action.to(torch.float32)
        else:
            action = torch.zeros(0)

        imgs = []
        for image_index in range(begin, end):
            img = Image.open(self.episode_images[ep][image_index])
            img = img.resize((self.res, self.res))
            img = transforms.ToTensor()(img)[:3]
            imgs.append(img)

        return DatasetItem(img=torch.stack(imgs, dim=0).float(), action=action)

    def __len__(self):
        if self.episodic:
            # Number of episodes
            return len(self.episode_images)
        else:
            # Number of available sequences of length self.sample_length
            return len(self.index2episode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--mode', choices=['train', 'val'], default='train')
    parser.add_argument('--sample_length', type=int, default=1)
    parser.add_argument('--episodic_on_train', action='store_true')
    parser.add_argument('--episodic_on_val', action='store_true')
    parser.add_argument('--use_actions', action='store_true')

    args = parser.parse_args()
    root = args.path
    mode = args.mode
    episodic_on_train = args.episodic_on_train
    episodic_on_val = args.episodic_on_val
    use_actions = args.use_actions
    sample_length = args.sample_length
    ds = EpisodesDataset(root, mode, sample_length=sample_length, res=64, episodic_on_val=episodic_on_val,
                         episodic_on_train=episodic_on_train, use_actions=use_actions)
    print('Length:', len(ds))
    element = ds[0]
    print('Shape:', element.img.size(), 'Mean:', element.img.mean())
