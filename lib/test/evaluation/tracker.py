import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import torch
from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
from lib.train.data.transformation_Tnf import SynthPairTnf,GeometricTnf
from lib.train.actors.geotnf.point_tnf import PointTnf
from torch.autograd import Variable
from .testM import cross_correlation
import torchvision.transforms.functional as tvisf
from .sift import Splice
import random


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name
        self.visdom= True
        self.ana = False
        self.save_results=False
        self.geometric_model='affine'  # affine or hom


        self.out_h=480
        self.out_w=480
        self.pair_generation_tnf = SynthPairTnf(use_cuda=False,geometric_model=self.geometric_model,crop_factor=1.0,
                                                output_size=(self.out_h,self.out_w), padding_factor=1.0, occlusion_factor=0)

        ##loss
        grid_size=20
        use_cuda=False
        # define virtual grid of points to be transformed
        axis_coords = np.linspace(-1,1,grid_size)
        self.N = grid_size*grid_size
        X,Y = np.meshgrid(axis_coords,axis_coords)
        X = np.reshape(X,(1,1,self.N))
        Y = np.reshape(Y,(1,1,self.N))
        P = np.concatenate((X,Y),1)
        self.P = Variable(torch.FloatTensor(P),requires_grad=False)
        self.pointTnf = PointTnf(use_cuda=use_cuda)
        if use_cuda:
            self.P = self.P.cuda()
        self.L1_loss = torch.nn.L1Loss()

        env = env_settings()

        if self.run_id is None:
            self.results_dir = '{}/{}/{}/{}'.format(env.results_path, self.name, self.parameter_name,self.dataset_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, dataset_seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()
        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_
        tracker = self.create_tracker(params)
        frame_ids=[1]
        # Get init information
        num_data = dataset_seq[3]
        dataset_name=dataset_seq[2]
        ### 保存结果
        MSE_all = []
        SSIM_all = []
        NCC_all = []
        RMSE_all = []
        loss_grid_all=[]
        loss_l1_all=[]
        AUC_1 = []
        AUC_3 = []
        AUC_5 = []
        AUC_7 = []
        AUC_10 = []
        for id in range(num_data):
            #id = random.randint(1, num_data)
            if dataset_name=='DroneVehicle_r':
                [frame_list_template, template_anno, frame_list_search,
                 search_anno, theta_GT, frame_template, frame_search] = self.get_frames_norandom(dataset_name=dataset_seq[2],
                                                                                        seq_path=dataset_seq[1][id],
                                                                                        theta=dataset_seq[4][id])
            else:
                [frame_list_template, template_anno, frame_list_search,
                search_anno, theta_GT,frame_template,frame_search]=self.get_frames(dataset_name=dataset_seq[2],seq_path=dataset_seq[1][id],frame_ids=frame_ids)
            '''
            sift
            '''
            # start = time.perf_counter()
            # theta_sift,flag=Splice(frame_list_template,frame_list_search)
            # if not flag:
            #     continue
            # end = time.perf_counter()
            # sift_runtime=end-start

            init_info = template_anno
            output = self._track_sequence(tracker, image_template=frame_list_template,image_search=frame_list_search,init_info=init_info)

            if isinstance(output['time'][0], (dict, OrderedDict)):
                exec_time = sum([sum(times.values()) for times in output['time']])
                num_frames = len(output['time'])
            else:
                exec_time = sum(output['time'])
                num_frames = len(output['time'])



            ## 计算配准指标
            P = self.P.expand(1, 2, self.N)

            theta=output['target'][0].cpu()

            # theta=theta_sift

            # compute transformed grid points using estimated and GT tnfs
            if self.geometric_model == 'affine':
                P_prime = self.pointTnf.affPointTnf(theta, P)*240
                P_prime_GT = self.pointTnf.affPointTnf(theta_GT, P)*240
                #p_prime_inv = self.pointTnf.affPointTnf(theta_inv, P_prime_GT)
            elif self.geometric_model == 'hom':
                tensor_1 = torch.ones(1, 1)
                theta = torch.cat((theta, tensor_1), dim=1)
                P_prime = self.pointTnf.homPointTnf(theta, P)*240
                theta_GT=theta_GT.view(1,9)
                P_prime_GT = self.pointTnf.homPointTnf(theta_GT, P)*240
            elif self.geometric_model == 'tps':
                P_prime = self.pointTnf.tpsPointTnf(theta.unsqueeze(2).unsqueeze(3), P)
                P_prime_GT = self.pointTnf.tpsPointTnf(theta_GT, P)


            # compute MSE loss on transformed grid points
            loss_point=torch.sum(torch.pow(P_prime - P_prime_GT, 2), 1)
            RSME=(torch.sum(loss_point)/self.N).sqrt()
            loss_sum = loss_point.sqrt()    #L1
            count_10 = torch.sum(loss_sum < 10 ) / self.N
            count_7 = torch.sum(loss_sum < 7) / self.N
            count_5 = torch.sum(loss_sum < 5) / self.N
            count_3 = torch.sum(loss_sum < 3) / self.N
            count_1 = torch.sum(loss_sum < 1) / self.N
            loss_grid = torch.mean(loss_sum)            #平均像素损失
            loss_L1 = self.L1_loss(theta, theta_GT)       #单应性损失


            ##print指标
            print('gt', theta_GT)
            print('theta', theta)
            print('FPS: {}'.format(num_frames / exec_time))
            print('loss_grid: {}'.format(loss_grid))
            print('loss_L1: {}'.format(loss_L1))
            print('auc10: {}'.format(count_10))
            print('auc7: {}'.format(count_7))
            print('auc5: {}'.format(count_5))
            print('auc3: {}'.format(count_3))
            print('auc1: {}'.format(count_1))

            frame_template_chw = np.transpose(frame_list_template, (2, 0, 1))
            frame_search_chw = np.transpose(frame_list_search, (2, 0, 1))

            frame_template_Tnf_chw, frame_search_Tnf_chw, theta = self.pair_generation_tnf(frame_template=frame_search_chw,
                                                                                                frame_search=frame_template_chw,
                                                                                                theta_batch=theta.numpy())

            # warped_image_aff = self.affTnf(frame_search_chw.view, theta)
            frame_template_Tnf = np.transpose(frame_template_Tnf_chw, (1, 2, 0))
            frame_search_Tnf = np.transpose(frame_search_Tnf_chw, (1, 2, 0))

            NCC,MSE,SSIM = cross_correlation(frame_template_Tnf, frame_search_Tnf)

            ### 保存结果
            MSE_all.append(MSE)
            SSIM_all.append(SSIM)
            NCC_all.append((NCC[0][0]))
            AUC_1.append(count_1)
            AUC_3.append(count_3)
            AUC_5.append(count_5)
            AUC_7.append(count_7)
            AUC_10.append(count_10)
            RMSE_all.append(RSME)
            loss_grid_all.append(loss_grid)
            loss_l1_all.append(loss_L1)
            ##可视化
            if self.visdom:
                #变化图像
                #画图
                N_subplots = 6
                fig, axs = plt.subplots(1, N_subplots)
                axs[0].imshow(frame_search)
                axs[0].set_title('search')
                axs[1].imshow(frame_template)
                axs[1].set_title('template')
                axs[2].imshow(frame_search_Tnf)
                axs[2].set_title('tgt')
                axs[3].imshow(frame_template_Tnf)
                axs[3].set_title('aff')
                #axs[4].imshow(np.concatenate((frame_search_Tnf[:,:,:2],frame_list_search[:,:,0:1]),axis=-1))

                axs[4].imshow(frame_search_Tnf-frame_template_Tnf)
                axs[4].set_title('tgt-sourse')
                axs[5].imshow(np.concatenate((frame_search_Tnf[:,:,:2],frame_template_Tnf[:,:,0:1]),axis=-1))
                axs[5].set_title('tgt-aff')
                for i in range(N_subplots):
                    axs[i].axis('off')
                print('Showing results. Close figure window to continue')
                plt.show()
                key = cv.waitKey(1)
                if key == ord('q'):
                    break

        print('Dataset_name: {}'.format(dataset_seq[0].split('/')[-1]))
        print('Number: {}'.format(int(len(RMSE_all))))
        if self.save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = dataset_seq[0].split('/')[-1]
            base_results_path = os.path.join(self.results_dir, 'video_{}.txt'.format(video_name))
            name=[RMSE_all,loss_grid_all,loss_l1_all,AUC_10,AUC_7,AUC_5,AUC_3,AUC_1,SSIM_all,MSE_all,NCC_all]
            with open(base_results_path, 'w') as file:
                for i in range(11):
                    for j in range(len(RMSE_all)):
                        file.write(str(np.array(name[i])[j]))
                        file.write(' ')
                    file.write('\n')


        return np.mean(AUC_5)

    def _track_sequence(self, tracker, image_search,image_template, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target': [],
                  'tmp':[],
                  'infere':[],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        start_time = time.time()
        out = tracker.match(image_search,image_template, init_info)
        if out is None:
            out = {}
        _store_outputs(out, {'time': time.time() - start_time})
        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            #tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
            print('errr')
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")


    def get_frames(self,dataset_name=None,seq_path=None, frame_ids=None):
        # COCO is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        if dataset_name=='VTUAV':
            frame_template = self._get_image_vtuav(seq_path,type='template')
            frame_search = self._get_image_vtuav(seq_path,type='search')
        elif dataset_name == 'VEDAI':
            frame_template = self._get_image_vedai(seq_path,type='template')
            frame_search = self._get_image_vedai(seq_path,type='search')
        elif dataset_name == 'DroneVehicle':
            frame_template = self._get_image_dronevehicle(seq_path,type='template')
            frame_template = frame_template[100:612, 100:740]
            frame_search = self._get_image_dronevehicle(seq_path,type='search')    #ir
            frame_search = frame_search[100:612, 100:740]


        target_size = (self.out_w, self.out_h)
        frame_template = cv.resize(frame_template, target_size)
        frame_search = cv.resize(frame_search, target_size)

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
        ####参数
        AFF={
                'random_t_hom': 0.2, #hom
                'rot_angle': 15,  # 旋转
                'sh_angle': 15,  # 剪切
                'lambda_1' : 0.6,  # 缩放（-1，1）
                'lambda_2': 0.6,
                'tx': 0.15 , # 平移(0,1)
                'ty': 0.15}

        rot_angle = (np.random.rand(1) - 0.5) * 2 * (
                    np.pi / 180 * AFF['rot_angle'])  # between -np.pi/60 and np.pi/60  60度
            #
        sh_angle = (np.random.rand(1) - 0.5) * 2 * (
                    np.pi / 180 * AFF['sh_angle'])  # between -np.pi/6 and np.pi/6   18度
            # 拉伸与压缩
        lambda_1 = 1 + (2 * np.random.rand(1) - 1) * AFF['lambda_1']  # between 0.9 and 1.1
        lambda_2 = lambda_1
            # 平移
        tx = (2 * np.random.rand(1) - 1) * AFF['tx']  # between -1 and 1
        ty = (2 * np.random.rand(1) - 1) * AFF['ty']

        R_sh = np.array([[np.cos(sh_angle[0]), -np.sin(sh_angle[0])],
                         [np.sin(sh_angle[0]), np.cos(sh_angle[0])]])

        R_alpha = np.array([[np.cos(rot_angle[0]), -np.sin(rot_angle[0])],
                            [np.sin(rot_angle[0]), np.cos(rot_angle[0])]])

        D = np.diag([lambda_1[0], lambda_2[0]])
        A = R_alpha @ R_sh.transpose() @ D @ R_sh

        if self.geometric_model == 'affine':
            theta = np.array([A[0, 0], A[0, 1], tx[0], A[1, 0], A[1, 1], ty[0]])# 6 个参数
        elif self.geometric_model=='hom':
            vx = (2 * np.random.rand(1) - 1) * AFF['random_t_hom']
            vy = (2 * np.random.rand(1) - 1) * AFF['random_t_hom']
            z = [1.000]
            theta = np.array([A[0, 0], A[0, 1], tx[0], A[1, 0], A[1, 1], ty[0], vx[0], vy[0], z[0]])


        frame_template_Tnf_chw,frame_search_Tnf_chw,theta = self.pair_generation_tnf(frame_template_chw,frame_search_chw, theta)

        frame_template_Tnf = np.transpose(frame_template_Tnf_chw, (1,2,0))
        frame_search_Tnf = np.transpose(frame_search_Tnf_chw, (1,2,0))
        # 绘制图像

        frame_list_search = frame_search.copy() #
        frame_list_template = frame_template_Tnf.copy()
        theta=torch.from_numpy(theta).type(torch.float32)
        return frame_list_template,template_anno,frame_list_search,search_anno, theta,frame_template,frame_search

    def get_frames_norandom(self, dataset_name=None, seq_path=None,theta=None):
        frame_template = self._get_image_dronevehicle(seq_path, type='template')
        frame_search = self._get_image_dronevehicle(seq_path, type='search')  # ir
        template_anno = self._get_anno(frame_template)
        search_anno = self._get_anno(frame_search)
        theta_list=theta.split( )
        theta_gt=np.array(theta_list[1:])
        theta_gt = np.array([np.fromstring(s, dtype='float32', sep=' ') for s in theta_gt]).reshape(-1)
        # 绘制图像

        frame_list_search = frame_search.copy()  #
        frame_list_template = frame_template.copy()
        theta_gt = torch.from_numpy(theta_gt).type(torch.float32)
        return frame_list_template, template_anno, frame_list_search, search_anno, theta_gt, frame_template, frame_search

    def _get_image_vtuav(self, seq_path,type):
        if type=='template':
            path=seq_path  #RGB
            img = self._read_image(os.path.join( path))
        elif type=='search':
            path_q = seq_path.split('rgb')
            basepath=path_q[0]
            path_part = seq_path.split('/')
            template_path='ir/'+path_part[-1]
            img = self._read_image(os.path.join(basepath+template_path))
        return img

    def _get_image_vedai(self, seq_path,type):
        if type=='search':
            path_q = seq_path.split('_')
            basepath = path_q[0]            #ir
            search_path = basepath + '_ir.png'
            img = self._read_image(os.path.join( search_path))
        elif type=='template':
            path=seq_path  #RGB
            img = self._read_image(os.path.join( path))
        return img
    def _get_image_dronevehicle(self, seq_path,type):
        if type=='search':
            path_q = seq_path.split('img')
            basepath = path_q[0] #ir
            search_path = basepath + 'imgi'
            img = self._read_image(os.path.join(search_path+path_q[-1]))
        elif type=='template':
            path=seq_path  #RGB
            img = self._read_image(os.path.join( path))
        return img
    def _get_anno(self, img):
        image_width,image_height,c = img.shape
        if image_width==self.out_w and image_height==self.out_h:
            left = (image_width - self.out_w) // 2
            top = (image_height - self.out_h) // 2
        else:
            print('size erorrs')
        anno=[left,top,self.out_w,self.out_h]
        anno=torch.Tensor(anno).type(torch.float32)
        return anno



