from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .geotnf.point_tnf import PointTnf

class TransformedGridLoss(nn.Module):
    def __init__(self, geometric_model='affine', use_cuda=True, grid_size=20):
        super(TransformedGridLoss, self).__init__()
        self.geometric_model = geometric_model
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
            self.P = self.P.cuda();

    def forward(self, theta=None,theta_GT=None):
        # expand grid according to batch size
        batch_size = theta.size()[0]
        P = self.P.expand(batch_size,2,self.N)
        '''
        # 初始化一个空的逆矩阵
        inverse_theta = torch.empty_like(theta)

        # 对每个 2*3 的子矩阵进行逆操作
        for i in range(batch_size):
            theta_matrix = theta[i].view(2, 3)
            theta_matrix_A=theta_matrix[:,:2]
            theta_matrix_xy=theta_matrix[:, 2]

            # 计算逆矩阵
            theta_matrix_A_inv = torch.pinverse(theta_matrix_A)
            theta_matrix_xy_f =-theta_matrix_xy.view(2,1)
            theta_matrix_xy_inv =torch.matmul(theta_matrix_A_inv,theta_matrix_xy_f)
            theta_inv = torch.cat((theta_matrix_A_inv, theta_matrix_xy_inv), dim=1).view(-1)
            inverse_theta[i] = theta_inv
        '''
        # compute transformed grid points using estimated and GT tnfs
        if self.geometric_model=='affine':
            P_prime = self.pointTnf.affPointTnf(theta,P)
            P_prime_GT = self.pointTnf.affPointTnf(theta_GT,P)
            #p_prime_inv= self.pointTnf.affPointTnf(theta_inv,P_prime_GT)
        elif self.geometric_model=='hom':
            P_prime = self.pointTnf.homPointTnf(theta,P)
            P_prime_GT = self.pointTnf.homPointTnf(theta_GT,P)
            #p_prime_inv = self.pointTnf.homPointTnf(theta_inv, P_prime_GT)
        elif self.geometric_model=='tps':
            P_prime = self.pointTnf.tpsPointTnf(theta.unsqueeze(2).unsqueeze(3),P)
            P_prime_GT = self.pointTnf.tpsPointTnf(theta_GT,P)
        # compute MSE loss on transformed grid points
        loss_sum = torch.sum(torch.pow(P_prime - P_prime_GT,2),1)
        loss_grid = torch.mean(loss_sum)


        return loss_grid