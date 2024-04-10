import torch
import torch.nn as nn
from ..util.drop_path import DropPathAllocator
class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        '''
            Args:
                x (torch.Tensor): (B, L, C), input tensor
            Returns:
                torch.Tensor: (B, L, C), output tensor
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 attn_pos_encoding_only=False):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if attn_pos_encoding_only:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_pos_encoding_only = attn_pos_encoding_only

    def forward(self, q, kv, q_ape, k_ape, attn_pos):
        '''
            Args:
                q (torch.Tensor): (B, L_q, C)
                kv (torch.Tensor): (B, L_kv, C)
                q_ape (torch.Tensor | None): (1 or B, L_q, C), absolute positional encoding for q
                k_ape (torch.Tensor | None): (1 or B, L_kv, C), absolute positional encoding for k
                attn_pos (torch.Tensor | None): (1 or B, num_heads, L_q, L_kv), untied positional encoding
            Returns:
                torch.Tensor: (B, L_q, C)
        '''
        B, q_N, C = q.shape
        kv_N = kv.shape[1]

        if self.attn_pos_encoding_only:
            assert q_ape is None and k_ape is None
            q = self.q(q).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv(kv).reshape(B, kv_N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            q = q + q_ape if q_ape is not None else q
            q = self.q(q).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = kv + k_ape if k_ape is not None else kv
            k = self.k(k).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(kv).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn * self.scale
        if attn_pos is not None:
            attn = attn + attn_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, q_N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_pos_encoding_only=False):
        super(CrossAttentionBlock, self).__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)

        self.drop_path = drop_path
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, kv, q_ape, k_ape, attn_pos):
        '''
            Args:
                q (torch.Tensor): (B, L_q, C)
                kv (torch.Tensor): (B, L_kv, C)
                q_ape (torch.Tensor | None): (1 or B, L_q, C), absolute positional encoding for q
                k_ape (torch.Tensor | None): (1 or B, L_kv, C), absolute positional encoding for k
                attn_pos (torch.Tensor | None): (1 or B, num_heads, L_q, L_kv), untied positional encoding
            Returns:
                torch.Tensor: (B, L_q, C)
        '''
        q = q + self.drop_path(self.attn(self.norm1_q(q), self.norm1_kv(kv), q_ape, k_ape, attn_pos))
        # q = q + self.drop_path(self.mlp(self.norm2(q)))

        return q

class ConcatenationBasedDecoder(nn.Module):
    def __init__(self, cross_attention_modules,
                 z_untied_pos_enc, x_untied_pos_enc,
                 rpe_bias_table, rpe_index):
        super(ConcatenationBasedDecoder, self).__init__()
        self.layers = nn.ModuleList(cross_attention_modules)
        self.z_untied_pos_enc = z_untied_pos_enc
        self.x_untied_pos_enc = x_untied_pos_enc
        if rpe_index is not None:
            self.register_buffer('rpe_index', rpe_index,False)
        self.rpe_bias_table = rpe_bias_table

    def forward(self, z, x, z_pos, x_pos):
        '''
            Args:
                z (torch.Tensor): (B, L_z, C)
                x (torch.Tensor): (B, L_x, C)
                z_pos (torch.Tensor | None): (1 or B, L_z, C)
                x_pos (torch.Tensor | None): (1 or B, L_x, C)
            Returns:
                torch.Tensor: (B, L_x, C)
        '''
        concatenated_pos_enc = None
        if z_pos is not None:
            concatenated_pos_enc = torch.cat((z_pos, x_pos), dim=1)

        attn_pos_enc = None
        if self.z_untied_pos_enc is not None:
            z_learned_pos_k = self.z_untied_pos_enc()
            x_learned_pos_q, x_learned_pos_k = self.x_untied_pos_enc()
            attn_pos_enc = x_learned_pos_q @ torch.cat((z_learned_pos_k, x_learned_pos_k), dim=1).transpose(-2, -1).unsqueeze(0)

        if self.rpe_bias_table is not None:
            if attn_pos_enc is not None:
                attn_pos_enc = attn_pos_enc + self.rpe_bias_table(self.rpe_index)
            else:
                attn_pos_enc = self.rpe_bias_table(self.rpe_index)

        for cross_attention in self.layers:
            #x = cross_attention(x, torch.cat((z, x), dim=1), x_pos, concatenated_pos_enc, attn_pos_enc)
            x = cross_attention(x,z, x_pos,z_pos,attn_pos_enc)
        return x

def build_CFA(d_model, dim_feedforward,
                                     dropout, activation,
                                     nhead,num_encoder_layers):

    traditional_positional_encoding_enabled=True

    mlp_ratio = 4
    drop_rate = dropout
    attn_drop_rate = 0
    qkv_bias = True
    untied_z_pos_enc = None
    untied_x_pos_enc = None
    rpe_index = None
    rpe_bias_table = None
    num_heads=nhead


    drop_path_rate = float(0.1)
    drop_path_allocator = DropPathAllocator(drop_path_rate)
    num_layers = 1 ##
    decoder_modules = []
    with drop_path_allocator:    # 初始化
        for index_of_decoder in range(num_layers):
            decoder_modules.append(
                CrossAttentionBlock(d_model, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=drop_path_allocator.allocate(),
                                attn_pos_encoding_only=not traditional_positional_encoding_enabled)
            )
            drop_path_allocator.increase_depth()
        CFA_model = ConcatenationBasedDecoder(decoder_modules, untied_z_pos_enc, untied_x_pos_enc,  rpe_bias_table, rpe_index)

    return CFA_model