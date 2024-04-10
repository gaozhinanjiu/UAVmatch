import os
from .base_image_dataset import BaseImageDataset
import torch
import random
from collections import OrderedDict
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
import glob
import numpy as np
import matplotlib.pyplot as plt
from lib.train.data.transformation_Tnf import SynthPairTnf

class DroneVehicleDataset(BaseImageDataset):
    """
    From coco dataset
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, min_area=None,out_size=1024,
                 split="rtest"):
        """
        args:
            root - path to coco root folder
            image_loader (jpeg4py_loader) - The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
            min_area - Objects with area less than min_area are filtered out. Default is 0.0
            split - 'train' or 'val'.
        """

        root = env_settings().DroneVehicle_norandom_dir if root is None else root
        super().__init__('DroneVehicle', root, image_loader)
        self.out_h=out_size
        self.out_w=out_size
        self.img_pth = os.path.join(root, '{}/'.format(split))
        self.txt=self.img_pth+'tp.txt'
        self.image_list = self._get_image_list(min_area=min_area)
        if data_fraction is not None:
            self.image_list = random.sample(self.image_list, int(len(self.image_list) * data_fraction))

        self.pair_generation_tnf = SynthPairTnf(use_cuda=False,geometric_model='affine',crop_factor=1.0,
                                                output_size=(self.out_h,self.out_w), padding_factor=1.0, occlusion_factor=0)


    def _get_image_list(self, min_area=None):
        root_img = self.img_pth + 'img'
        image_list = glob.glob(root_img + '/*')

        if min_area is not None:
            print('error!')
            # image_list = [a for a in image_list if self.coco_set.anns[a]['area'] > min_area]
        return image_list

    def get_sequence_list(self, min_area=None):
        theta=[]
        with open(self.txt, 'r') as file:
            # 逐行读取文件内容
            for line in file:
                theta.append(line)
        # 关闭文件
        file.close()
        root_img = self.img_pth + 'img'
        image_list = glob.glob(root_img + '/*')
        image_list =sorted(image_list)
        num_p=int(len(image_list)/4)
        image_list1 = image_list[:num_p]
        theta_list1 = theta[:num_p]
        image_list2 = image_list[num_p:2 * num_p]
        theta_list2 = theta[num_p:2 * num_p]
        image_list3 = image_list[2 * num_p:3 * num_p]
        theta_list3 = theta[2*num_p:3 * num_p]
        image_list4 = image_list[3 * num_p:]
        theta_list4 = theta[3*num_p:4 * num_p]

        if min_area is not None:
            print('error!')
            # image_list = [a for a in image_list if self.coco_set.anns[a]['area'] > min_area]
        return [['DroneVehicle001',image_list1,'DroneVehicle_r',num_p,theta_list1],
                ['DroneVehicle002',image_list2,'DroneVehicle_r',num_p,theta_list2],
                ['DroneVehicle003',image_list3,'DroneVehicle_r',num_p,theta_list3],
                ['DroneVehicle004',image_list4,'DroneVehicle_r',num_p,theta_list4]]
    '''
    def get_num_classes(self):
        return len(self.class_list)
    '''

    def get_name(self):
        return 'DroneVehicle'

    def has_class_info(self):
        return True

    def has_segmentation_info(self):
        return True
    '''
    def get_class_list(self):
        class_list = []
        for cat_id in self.cats.keys():
            class_list.append(self.cats[cat_id]['name'])
        return class_list
    
    
    def _build_im_per_class(self):
        im_per_class = {}
        for i, im in enumerate(self.image_list):
            class_name = self.cats[self.coco_set.anns[im]['category_id']]['name']
            if class_name not in im_per_class:
                im_per_class[class_name] = [i]
            else:
                im_per_class[class_name].append(i)

        return im_per_class

    '''
    def is_video_sequence(self):
        return False

    def get_num_sequences(self):
        return len(self.image_list)

    def get_images_in_class(self, class_name):
        return self.im_per_class[class_name]



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

    def _get_image(self, im_id,type):
        if type=='search':
            path=self.image_list[im_id] #A
            img = self.image_loader(os.path.join(self.img_pth, path))
        elif type=='template':
            path_part = self.image_list[im_id].split('/')
            template_path='t2/'+path_part[-1]
            img = self.image_loader(os.path.join(self.img_pth, template_path))
        return img

    def get_meta_info(self, im_id):
        try:
            cat_dict_current = self.cats[self.coco_set.anns[self.image_list[im_id]]['category_id']]
            object_meta = OrderedDict({'object_class_name': cat_dict_current['name'],
                                       'motion_class': None,
                                       'major_class': cat_dict_current['supercategory'],
                                       'root_class': None,
                                       'motion_adverb': None})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def get_class_name(self, im_id):
        cat_dict_current = self.cats[self.coco_set.anns[self.image_list[im_id]]['category_id']]
        return cat_dict_current['name']

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        # COCO is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.


        frame_template = self._get_image(seq_id,type='template')
        frame_search = self._get_image(seq_id,type='search')


        template_anno=self._get_anno(frame_template)
        search_anno = self._get_anno(frame_search)
##
        # 图像变化到CHW空间
        frame_template_chw = np.transpose(frame_template, (2, 0, 1))
        frame_search_chw = np.transpose(frame_search, (2, 0, 1))
        '''
        # 绘制图像
        fig, axs = plt.subplots(1, 2)

        # 在第一个子图中绘制图像1
        axs[0].imshow(frame_search_Tnf)
        axs[0].axis('off')  # 关闭坐标轴
        axs[0].set_title('template')

        # 在第二个子图中绘制图像2
        axs[1].imshow(frame_search)
        axs[1].axis('off')  # 关闭坐标轴
        axs[1].set_title('Search')

        # 调整子图之间的间距
        plt.tight_layout()
        # 显示图像
        plt.show()
    '''
        '''
        生成单应性变化参数H
        '''
        rot_angle = (np.random.rand(1) - 0.5) * 2 * np.pi / 180 * 120 ;  # between -np.pi/60 and np.pi/60  60度
        #
        sh_angle = (np.random.rand(1) - 0.5) * 2 * np.pi /180   * 20;  # between -np.pi/6 and np.pi/6   18度
        # 拉伸与压缩
        lambda_1 = 1 + (2 * np.random.rand(1) - 1) * 0.2;  # between 0.9 and 1.1
        lambda_2 = lambda_1;
        # 平移
        tx = (2 * np.random.rand(1) - 1) * 1;  # between -1 and 1
        ty = (2 * np.random.rand(1) - 1) * 1;

        R_sh = np.array([[np.cos(sh_angle[0]), -np.sin(sh_angle[0])],
                         [np.sin(sh_angle[0]), np.cos(sh_angle[0])]])

        R_alpha = np.array([[np.cos(rot_angle[0]), -np.sin(rot_angle[0])],
                            [np.sin(rot_angle[0]), np.cos(rot_angle[0])]])

        D = np.diag([lambda_1[0], lambda_2[0]])
        A = R_alpha @ R_sh.transpose() @ D @ R_sh

        theta_aff = np.array([A[0, 0], A[0, 1], tx[0], A[1, 0], A[1, 1], ty[0]]) #6 个参数
        '''
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[0, ...] for _ in frame_ids]

        object_meta = self.get_meta_info(seq_id)
        '''
        frame_template_Tnf_chw,frame_search_Tnf_chw,theta_aff = self.pair_generation_tnf(frame_template_chw,frame_search_chw, theta_aff)

        frame_template_Tnf = np.transpose(frame_template_Tnf_chw, (1,2,0))
        frame_search_Tnf = np.transpose(frame_search_Tnf_chw, (1,2,0))


        # 绘制图像

        frame_list_search = [frame_search.copy() for _ in frame_ids]
        frame_list_template = [frame_template_Tnf.copy() for _ in frame_ids]

        return frame_list_template,template_anno,frame_list_search,search_anno, theta_aff
