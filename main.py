"""
This is the main script to evaluate a video.
Given a checkpoint to load and a video folder, computes
negative log-likelihood scores for each clip in the video.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from os.path import basename
from tqdm import tqdm

from dataset import SHANGHAITECH
from model import AND
from model.functional import autoregression_nll

plt.ion()


def parse_arguments():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='The path of stored model weights', metavar='')
    parser.add_argument('--video_path', type=str,
                        help='The path of the video to evaluate', metavar='')
    return parser.parse_args()


class ScoreAccumulator:
    """
    Provides frame-level score from clip-level
    scores, by accumulating and averaging.
    """

    def __init__(self, time_steps):
        """
        Class Constructor.

        Parameters
        ----------
        time_steps: int
            the number of time steps composing each clip.
        """

        # Initialize buffer and counts
        self._buffer = np.zeros(shape=(time_steps,), dtype=np.float32)
        self._counts = np.zeros(shape=(time_steps,), dtype=np.uint8)

    def push(self, clip_score):
        """
        Pushes the score of the last clip to be summed to all elements
        of the buffer.

        Parameters
        ----------
        clip_score: float
            the score of the last clip.
        """

        # Update buffer and counts
        self._buffer += clip_score
        self._counts += 1

    def get_next(self):
        """
        Returns the result of the oldest frame (first in buffer)
        and rolls buffer and counts.

        Returns
        -------
        ret_score: float
            the score of the first frame in buffer,
            computed as the mean of each clip in which it
            belongs.
        """

        # Save first in buffer
        ret_score = self._buffer[0] / self._counts[0]

        # Roll time backwards D=
        self._buffer = np.roll(self._buffer, shift=-1)
        self._counts = np.roll(self._counts, shift=-1)

        # Zero out final frame
        self._buffer[-1] = 0
        self._counts[-1] = 0

        return ret_score

    def results_left(self):
        """
        Returns how many frames are still in memory to be evaluated.

        Returns
        -------
        int
            the number of frames still to be returned.
        """

        return np.sum(self._counts != 0).astype(np.int32)


def evaluate_video(model, data_loader):
    """
    Evaluates scores for each 16-frames clip in a given video.

    Parameters
    ----------
    model: torch.nn.Module
        the AND model scoring clips.
    data_loader: DataLoader
        DataLoader object providing clips.

    Returns
    -------
    tuple
        frames: np.array
            all processed frames, has shape=(n_frames, h, w, c)
        scores: np.array
            scores for each frame, has shape=(n_frames,)
    """

    # Get dataset shape
    c, t, h, w = data_loader.dataset.shape

    # Instantiate score accumulator
    scores_acc = ScoreAccumulator(time_steps=t)

    # Instantiate results array
    scores = np.zeros(shape=(len(data_loader) + t - 1))
    frames = np.zeros(shape=(len(data_loader), h, w, c))

    # Loop over video clips
    for clip_idx, (x, _) in tqdm(enumerate(data_loader),
                                 desc='Evaluating video: {}'.format(data_loader.dataset.video_path)):

        # Turn to Variable
        x = Variable(x, volatile=True).cuda()

        # Feed the model
        x_rec, z, z_dist = model(x)

        # Evaluate the negative log-likelihood of the clip
        nll, _ = autoregression_nll(z, z_dist, autoregression_bins=100)

        # Feed scores accumulator
        scores_acc.push(nll.data.cpu())

        # Get the score of the first frame in buffer
        scores[clip_idx] = scores_acc.get_next()

        # Save frame (for later visualization)
        frames[clip_idx] = x[0, :, 0].data.cpu().numpy().transpose(1, 2, 0)

    # Get last results from accumulator
    while True:
        left = scores_acc.results_left()
        if left == 0:
            break
        scores[-left] = scores_acc.get_next()

    return frames, scores


def plot_video_score(video_id, frames, scores):
    """
    Plots the video and the regularity score.

    Parameters
    ----------
    video_id: str
        the id of the video.
    frames: np.array
        Array holding video frames.
        Has shape=(n_frames, h, w, c).
    scores: np.array
        Array holding scores.
        Has shape=(n_frames,).
    """

    # Initialize subplots
    fig, (ax0, ax1) = plt.subplots(2, 1)
    fig.suptitle('Test video: {}'.format(video_id))

    # Loop over frames and scores
    for idx, frame in enumerate(frames):

        ax0.cla()
        ax0.imshow(frame)
        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1.cla()
        ax1.plot(scores[:idx+1])
        ax1.set_xlim(0, len(scores))
        ax1.set_ylim(np.min(scores), np.max(scores))
        ax1.set_ylabel('NLP')
        ax1.set_xlabel('Frames')
        plt.pause(0.02)


def main():
    """
    Main function.
    """

    # Parse command line arguments.
    args = parse_arguments()

    # Get dataset
    dataset = SHANGHAITECH()
    dataset.set_video(args.video_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Get model
    model = AND(input_shape=dataset.shape,
                code_length=64,
                autoregression_bins=100).cuda()
    model.eval()

    # Load checkpoint
    model.load_state_dict(torch.load(args.checkpoint_path))

    # Evaluate all video
    frames, scores = evaluate_video(model, data_loader)

    # Visualize
    video_id = basename(args.video_path)
    plot_video_score(video_id, frames, scores)


# Entry point
if __name__ == '__main__':
    main()
