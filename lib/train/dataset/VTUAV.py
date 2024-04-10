import os
from .base_video_dataset  import BaseVideoDataset
from lib.train.data.image_loader import opencv_loader,jpeg4py_loader
import torch
import csv
import random
import cv2
from collections import OrderedDict
import pandas
import numpy as np
from lib.train.admin import env_settings
from ..data.transformation_Tnf import SynthPairTnf
import matplotlib.pyplot as plt
class VTUAV(BaseVideoDataset):

    def __init__(self, settings,root=None, image_loader=opencv_loader, split="train", modality=None,out_w=640,out_h=640):
        """
        args:
            root - path to uav root folder
            image_loader (jpeg4py_loader) - The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
            split - 'train' or 'val'.
            modality RGBT
        """
        if modality is None:
            raise ValueError('Unknown modality mode.')
        else:
            self.modality = modality

        super().__init__('UAVRGBT', root, image_loader)
        self.root_i = root
        # all folders inside the root
        self.sequence_list = self._get_sequence_list()
        self.out_w=out_w
        self.out_h=out_h
        self.settings=settings
        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            ltr_path = os.path.join(self.root_i)
            if split == 'train':
                file_path = os.path.join(ltr_path, 'ST_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'ST_val_split.txt')
            else:
                raise ValueError('Unknown split name.')
            sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()

        # self.seq_ids = seq_ids
        self.init_idx = np.load(os.path.abspath(os.path.join(env_settings().VTUAV_dir, 'init_frame.npy')),allow_pickle=True).item()
        self.sequence_list = sequence_list
        self.root = root+'train'
        self.root_i = root+'train'
        self.pair_generation_tnf = SynthPairTnf(use_cuda=False,geometric_model=self.settings.AFF.type,crop_factor=1.0,
                                                output_size=(self.out_h,self.out_w), padding_factor=1.0, occlusion_factor=0)



    def get_name(self):
        return 'VTUAV'
    def is_video_sequence(self):
        return True
    def has_class_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class


    def _get_parent_classList(self):
        project_path = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(project_path, '../data_specs/parent_class_imagenet_extended.txt')

        # load the parent class file -> refer to the imagenet website for the list of parent classes
        f = open(file_path)
        major_classes = list(csv.reader(f))
        f.close()

        parent_classes = [cls[0] for cls in major_classes]
        return parent_classes

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]


    def _get_sequence_list(self):

        return os.listdir(self.root)

    def _read_bb_anno(self, seq_path):
        if self.modality in ['RGB', 'RGBT']:
            bb_anno_file = os.path.join(seq_path, "rgb.txt")
            gt = np.loadtxt(bb_anno_file).astype(np.float32)
        elif self.modality in ['T']:
            bb_anno_file = os.path.join(seq_path, "ir.txt")
            gt = np.loadtxt(bb_anno_file).astype(np.float32)
        else:
            raise ValueError('Unknown modality mode.')

        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover > 0)

        return target_visible

    def _get_sequence_path_i(self, seq_id):
        return os.path.join(self.root_i, self.sequence_list[seq_id])

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)

        visible = valid

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, modality, frame_id):
        seq_name = seq_path.split('/')[-1]
        if seq_name in self.init_idx:
            init_idx = self.init_idx[seq_name]
        else:
            init_idx = 0
        nz = 6
        return os.path.join(seq_path, modality, str(frame_id * 10 + init_idx).zfill(nz) + '.jpg')  # frames start from 1

    def _get_frame(self, seq_path, modality, frame_id):
        '''
        if modality== 'T':
            pic=self.image_loader(self._get_frame_path(seq_path, modality, frame_id))
            pic_g = pic[:,:,1]
            equ = cv2.equalizeHist(pic_g)
            pic[:,:,0]=equ
            pic[:, :, 1] = equ
            pic[:, :, 2] = equ
            return pic
        '''
        return self.image_loader(self._get_frame_path(seq_path, modality, frame_id))

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class']

    def _get_anno(self, img):
        image_width,image_height,c = img.shape
        if image_width==self.out_w and image_height==self.out_h:
            left = (image_width - self.out_w) // 2
            top = (image_height - self.out_h) // 2
        else:
            print('size erorrs')
        anno=[left,top,self.out_w,self.out_h]
        anno=[torch.Tensor(anno).type(torch.float32)]
        return anno

    def get_frames(self, seq_id, frame_ids, anno=None):

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
            # anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids] + [value[f_id, ...].clone() for f_id in frame_ids]


        if self.modality in ['RGB']:
            seq_path = self._get_sequence_path(seq_id)
            frame_list = [self._get_frame(seq_path, 'rgb', f_id) for f_id in frame_ids]

        elif self.modality in ['T']:
            seq_path_i = self._get_sequence_path_i(seq_id)
            frame_list = [self._get_frame(seq_path_i, 'ir', f_id) for f_id in frame_ids]
        elif self.modality in ['RGBT']:
            seq_path = self._get_sequence_path(seq_id)
            seq_path_i = self._get_sequence_path_i(seq_id)
            target_size = (self.out_w, self.out_h)
            for f_id in frame_ids:
                frame_template=self._get_frame(seq_path, 'rgb',f_id)
                frame_search=self._get_frame(seq_path_i, 'ir',f_id)
            frame_template=cv2.resize(frame_template,target_size)
            frame_search = cv2.resize(frame_search, target_size)

            # 图像变化到CHW空间
            frame_template_chw = np.transpose(frame_template, (2, 0, 1))
            frame_search_chw = np.transpose(frame_search, (2, 0, 1))

            '''
            生成单应性变化参数H
            '''

            rot_angle = (np.random.rand(1) - 0.5) * 2 * (np.pi / 180 * self.settings.AFF.rot_angle)  # between -np.pi/60 and np.pi/60  60度
                #
            sh_angle = (np.random.rand(1) - 0.5) * 2 * (np.pi / 180 * self.settings.AFF.sh_angle)  # between -np.pi/6 and np.pi/6   18度
                # 拉伸与压缩
            lambda_1 = 1 + (2 * np.random.rand(1) - 1) * self.settings.AFF.lambda_1  # between 0.9 and 1.1
            lambda_2 = lambda_1
                # 平移
            tx = (2 * np.random.rand(1) - 1) * self.settings.AFF.tx  # between -1 and 1
            ty = (2 * np.random.rand(1) - 1) * self.settings.AFF.ty

            R_sh = np.array([[np.cos(sh_angle[0]), -np.sin(sh_angle[0])],
                             [np.sin(sh_angle[0]), np.cos(sh_angle[0])]])

            R_alpha = np.array([[np.cos(rot_angle[0]), -np.sin(rot_angle[0])],
                                [np.sin(rot_angle[0]), np.cos(rot_angle[0])]])

            D = np.diag([lambda_1[0], lambda_2[0]])
            A = R_alpha @ R_sh.transpose() @ D @ R_sh

            # 6 个参数
            if self.settings.AFF.type == 'affine':
                theta = np.array([A[0, 0], A[0, 1], tx[0], A[1, 0], A[1, 1], ty[0]])  # 6 个参数
            elif self.settings.AFF.type=='hom':
                vx = (2 * np.random.rand(1) - 1) * self.settings.AFF.random_t_hom
                vy = (2 * np.random.rand(1) - 1) * self.settings.AFF.random_t_hom
                z=[1.000]
                theta = np.array([A[0, 0], A[0, 1], tx[0], A[1, 0], A[1, 1], ty[0],vx[0],vy[0],z[0]])

            '''
            anno_frames = {}
            for key, value in anno.items():
                anno_frames[key] = [value[0, ...] for _ in frame_ids]

            object_meta = self.get_meta_info(seq_id)
            '''
            frame_template_Tnf_chw, frame_search_Tnf_chw, theta = self.pair_generation_tnf(frame_template_chw,
                                                                                               frame_search_chw,
                                                                                               theta)

            frame_template_Tnf = np.transpose(frame_template_Tnf_chw, (1, 2, 0))
            frame_search_Tnf = np.transpose(frame_search_Tnf_chw, (1, 2, 0))


            # 绘制图像
            '''
            fig, axs = plt.subplots(1, 4)
            # 在第一个子图中绘制图像1
            axs[0].imshow(frame_template_Tnf)
            axs[0].axis('off')  # 关闭坐标轴
            axs[0].set_title('template_TNF')

            # 在第二个子图中绘制图像2
            axs[1].imshow(frame_search_Tnf)
            axs[1].axis('off')  # 关闭坐标轴
            axs[1].set_title('search')

            axs[2].imshow(frame_template)
            axs[2].axis('off')  # 关闭坐标轴
            axs[2].set_title('template1')

            # 在第二个子图中绘制图像2
            axs[3].imshow(frame_search)
            axs[3].axis('off')  # 关闭坐标轴
            axs[3].set_title('Search1')
            # 调整子图之间的间距
            plt.tight_layout()
            # 显示图像
            plt.show()
            '''

            frame_list_search = [frame_search_Tnf.copy() for _ in frame_ids]
            frame_list_template = [frame_template_Tnf.copy() for _ in frame_ids]
            template_anno = self._get_anno(frame_template_Tnf)
            search_anno = self._get_anno(frame_search_Tnf)

        return frame_list_template, template_anno, frame_list_search,search_anno, theta



""" Test
def main():
    uavpath='/home/ldd/Videos/UAVRGBT'
    uavRGBT=UAVRGBT(uavpath, split='train', modality = 'RGBT')

if __name__ == '__main__':
    main()
    
"""