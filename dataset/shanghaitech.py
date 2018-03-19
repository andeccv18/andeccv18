"""
This file holds the class modeling the ShanghaiTech dataset.
"""

import torch
import numpy as np
import skimage.io as io
from skimage.transform import resize
from torch.utils.data import Dataset
from os.path import join
from glob import glob


class ToFloatTensor3D(object):
    """ This transform turns a numpy tensor to a float tensor. """

    def __call__(self, sample):
        """
        Performs the transform.

        Parameters
        ----------
        sample: tuple
            x: np.array
                array holding a clip, having shape=(t, h, w, c)
            y: np.array
                array holding a clip, having shape=(t, h, w, c)

        Returns
        -------
        sample: tuple
            x: FloatTensor
                array holding a clip, having shape=(t, c, h, w)
            y: FloatTensor
                array holding a clip, having shape=(t, c, h, w)
        """

        x, y = sample

        # Swap color axis
        # (T x H x W x C) -> (C, T, H, W)
        x = x.transpose(3, 0, 1, 2).astype(np.float32)
        y = y.transpose(3, 0, 1, 2).astype(np.float32)

        # Normalize
        x = x / 255.
        y = y / 255.

        return torch.from_numpy(x), torch.from_numpy(y)


class SHANGHAITECH(Dataset):
    """
    This class models the dataset.
    """

    def __init__(self):
        """
        Class constructor.
        """

        super(SHANGHAITECH, self).__init__()

        # Transform
        self.transform = ToFloatTensor3D()

        # Initialize length to zero
        self.cur_len = 0

        # List of frames
        self.frames_list = None

        # Initialize video path
        self.video_path = None

    def set_video(self, video_path):
        """
        Sets up the video to evaluate.

        Parameters
        ----------
        video_path: str
            the path in which image files are stored.
        """

        self.video_path = video_path

        # Get shape
        c, t, h, w = self.shape

        # Initialize frame list
        self.frames_list = sorted(glob(join(video_path, '*.jpg')))

        # Set the length
        self.cur_len = len(self.frames_list) - t + 1

    @property
    def shape(self):
        """
        Returns the shape of each dataset sample.

        Returns
        -------
        tuple
            the shape of each sample (c, t, h, w).
        """

        return 3, 16, 256, 512

    def __len__(self):
        """
        Returns the number of samples for the dataset.

        Returns
        -------
        int
            the length of the dataset.
        """

        return self.cur_len

    def __getitem__(self, i):
        """
        Loads and returns a clip from file, given an index.

        Parameters
        ----------
        i: int
            the index of the clip to load.

        Returns
        -------
        sample: tuple
            x: input clip as FloatTensor having shape=(c, t, h, w).
            y: target clip as FloatTensor having shape=(c, t, h, w).
        """

        # Index check
        assert i < self.cur_len, 'Invalid index {}. Something went wrong.'.format(i)

        # Get dataset shape
        c, t, h, w = self.shape

        # Load clip
        frames = []
        for offset in range(0, t):
            frame = io.imread(self.frames_list[i + offset])
            frame = resize(frame, output_shape=(h, w), mode='constant', preserve_range=True)
            frames.append(frame)
        frames = np.stack(frames, axis=0)
        sample = frames, frames

        # Apply transform
        if self.transform:
            sample = self.transform(sample)

        return sample
