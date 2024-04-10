import torch
import torch.nn as nn
import numpy as np
class Cross_Mod(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ = in_ch
        self.out_ = out_ch
        self.sconv = nn.Conv2d(in_ch, out_ch, 3, 1,1)  # 进行特征图尺寸减半
        self.conv_cat = nn.Conv2d(2 * out_ch, 2 * out_ch, 3, 1, 1)  # 用于拼接后的卷积核
        self.convs1 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, 1, 1)
                                   , nn.ReLU())
        self.convs2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, 1, 1)
                                   , nn.ReLU())

    def forward(self, vi,ir):
        b, hw, c = ir.size()
        ir=ir.view(b,int(np.sqrt(hw)),int(np.sqrt(hw)),c).transpose(1,3)
        vi=vi.view(b,int(np.sqrt(hw)),int(np.sqrt(hw)),c).transpose(1,3)
        ir = self.sconv(ir)
        vi = self.sconv(vi)
        img_cat = torch.cat((vi, ir), dim=1)

        img_conv = self.conv_cat(img_cat)

        # 平均拆分拼接后的特征
        split_ir = img_conv[:, self.out_:]  # 前一半特征
        split_vi = img_conv[:, :self.out_]  # 后一半特征

        # IR图像处理流程
        ir_1 = self.convs1(split_ir)
        ir_2 = self.convs2(split_ir)
        ir_mul = torch.mul(ir, ir_1)
        ir_add = ir_2 + ir_mul
        ir_add=ir_add.transpose(1,3).reshape(b,-1,c)

        # VI图像处理流程
        vi_1 = self.convs1(split_vi)
        vi_2 = self.convs2(split_vi)
        vi_mul = torch.mul(vi, vi_1)
        vi_add = vi_2 + vi_mul
        vi_add=vi_add.transpose(1,3).reshape(b,-1,c)
        return vi_add,ir_add


def build_Cross_model(input,output):


    Cross_model = Cross_Mod(in_ch=input,out_ch=output)

    return Cross_model