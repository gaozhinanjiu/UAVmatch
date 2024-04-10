import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
import glob

class VTUAVDataset(BaseDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """
    def __init__(self, split):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        if split == 'test' or split == 'val':
            self.base_path = os.path.join(self.env_settings.VTUAV)
        else:
            self.base_path = os.path.join(self.env_settings.VTUAV)

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):

        frame_list=glob.glob(sequence_name + '/rgb/*')
        num=len(frame_list)
        frame_list_roll=[]
        for i in range(int(num/50)):
            frame_list_roll.append(frame_list[50*i])

        return [sequence_name, frame_list_roll, 'VTUAV', int(num/50)]
    '''
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'VTUAV', ground_truth_rect.reshape(-1, 4))

    '''
    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        sequence_list = glob.glob(self.base_path+'*')
        '''
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        if split == 'ltrval':
            with open('{}/got10k_val_split.txt'.format(self.env_settings.dataspec_path)) as f:
                seq_ids = f.read().splitlines()

            sequence_list = [sequence_list[int(x)] for x in seq_ids]
        '''
        return sequence_list