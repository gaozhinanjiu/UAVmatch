
import torch.nn as nn
from .build_cross import build_Cross_model

class mulimodel(nn.Module):
    def __init__(self, d_model, dim_feedforward,
                    dropout, activation,
                    nhead,num_encoder_layers):
        super().__init__()
        self.in_ = d_model
        self.out_ = d_model

        self.cross_model0=build_Cross_model(self.in_,self.out_)
        self.cross_model1=build_Cross_model(self.in_,self.out_)
        self.cross_model2=build_Cross_model(self.in_,self.out_)


    def forward(self,src_flatten_z,src_flatten_x,lvl_pos_embed_flatten_z,lvl_pos_embed_flatten_x):
        src_flatten_z[0], src_flatten_x[0] = self.cross_model0(src_flatten_z[0],src_flatten_x[0])
        src_flatten_z[1], src_flatten_x[1] = self.cross_model1(src_flatten_z[1], src_flatten_x[1])
        src_flatten_z[2], src_flatten_x[2] = self.cross_model2(src_flatten_z[2], src_flatten_x[2])

        return src_flatten_z, src_flatten_x





def build_mulimodel(d_model, dim_feedforward,
                    dropout, activation,
                    nhead,num_encoder_layers):

    muli_model = mulimodel(d_model, dim_feedforward,
                    dropout, activation,
                    nhead,num_encoder_layers)

    return muli_model