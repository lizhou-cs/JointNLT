import os

import numpy as np
import re

from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class TNL2KDataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.tnl2k_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        attributes_path = '{}/{}/attributes.txt'.format(self.base_path, sequence_name)

        attributes_rect = load_text(str(attributes_path), delimiter=',', dtype=np.float64)
        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        # full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        # out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(self.base_path, class_name, sequence_name)
        # out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')
        nlp_path = '{}/{}/language.txt'.format(self.base_path, sequence_name)
        nlp_rect = load_text(str(nlp_path), delimiter=',', dtype=str)
        nlp_rect = str(nlp_rect)
        # print(f'nlp_rect  type: {type(nlp_rect)}, value:{nlp_rect}')
        # target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/imgs'.format(self.base_path, sequence_name)
        frames_list = os.listdir(frames_path)
        regex_end = re.compile(r"[0-9]*")
        try:
            frames_list.sort(key=lambda x: int(re.findall(regex_end, x)[0]))
        except:
            print(frames_path)
        # absolute path
        f_list = [os.path.join(frames_path, p) for p in frames_list]
        return Sequence(sequence_name, f_list, 'tnl2k', ground_truth_rect.reshape(-1, 4),
                        object_class=None, target_visible=None, language_query=nlp_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        return os.listdir(self.base_path)

