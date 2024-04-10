
import torch
from torch import nn
import torch.nn.functional as F
from lib.utils.misc import NestedTensor

from lib.models.stark_dual_deform.backbone.backbone import build_backbone
from lib.models.stark_dual_deform.transform.deformable_transformer import build_deforamble_transformer
from lib.models.stark_dual_deform.head.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh


class match(nn.Module):
    """ This is the base class for Transformer Tracking """
    def __init__(self, backbone_template,backbone_search, transformer,box_head, num_queries,
                 aux_loss=False, head_type="CORNER",num_feature_levels=3):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone_template = backbone_template
        self.backbone_search  = backbone_search
        self.transformer = transformer
        self.box_head = box_head
        self.num_queries = num_queries
        hidden_dim = transformer.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 8, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2) # object queries
        #num_pred = (transformer.decoder.num_layers + 1)
        #self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])

        ###

        ## backbone prjconv
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone_template.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone_template.num_strides_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.bottleneck_template = nn.ModuleList(input_proj_list)
        else:
            self.bottleneck_template = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_template.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])


        if num_feature_levels > 1:
            num_backbone_outs = len(backbone_template.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone_template.num_strides_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.bottleneck_search = nn.ModuleList(input_proj_list)
        else:
            self.bottleneck_search = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_template.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )]) # the bottleneck layer
        self.aux_loss = aux_loss
        self.head_type = head_type
        self.temperature=0.1  #参数


    def forward(self, img=None, seq_dict=None, mode="backbone", run_box_head=True, run_cls_head=False):
        if mode == "backbone_template":
            return self.forward_backbone_template(img)
        elif mode == "backbone_search":
            return self.forward_backbone_search(img)
        elif mode == "transformer":
            return self.forward_transformer(seq_dict, run_box_head=run_box_head, run_cls_head=run_cls_head)
        else:
            raise ValueError

    def forward_backbone(self, input: NestedTensor):
        """The input type is NestedTensor, which consists of:
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(input, NestedTensor)
        # Forward the backbone
        output_back, pos = self.backbone(input)  # features & masks, position embedding for the search
        # Adjust the shapes
        return self.adjust(output_back, pos)

    def forward_backbone_template(self, input: NestedTensor):
        """The input type is NestedTensor, which consists of:
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(input, NestedTensor)
        # Forward the backbone
        output_back, pos = self.backbone_template(input)  # features & masks, position embedding for the search
        # Adjust the shapes
        srcs = []
        masks = []
        poss=[]
        for lvl, (output_back1, pos1) in enumerate(zip(output_back,pos)):
            feat_vec, mask_vec,  pos_embed_vec=self.adjust_template(output_back1, pos1,lvl)
            srcs.append(feat_vec)
            masks.append(mask_vec)
            poss.append(pos_embed_vec)
        return {"feat": srcs, "mask": masks, "pos": poss}

    def forward_backbone_search(self, input: NestedTensor):
        """The input type is NestedTensor, which consists of:
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(input, NestedTensor)
        # Forward the backbone
        output_back, pos = self.backbone_search(input)  # features & masks, position embedding for the search
        # Adjust the shapes
        srcs = []
        masks = []
        poss = []
        for lvl, (output_back1, pos1) in enumerate(zip(output_back, pos)):
            feat_vec, mask_vec, pos_embed_vec = self.adjust_template(output_back1, pos1,lvl)
            srcs.append(feat_vec)
            masks.append(mask_vec)
            poss.append(pos_embed_vec)
        return {"feat": srcs, "mask": masks, "pos": poss}

    def forward_transformer(self, seq_dict, run_box_head=True, run_cls_head=False):
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        # Forward the transformer encoder and decoder

        output_embed, init_reference, inter_references = self.transformer(seq_dict["feat"], seq_dict["mask"],
                                                                                    seq_dict["pos"],self.query_embed.weight)

        tmp = self.bbox_embed(output_embed)


        reference = inter_references
        reference = self.inverse_sigmoid(reference)
        if reference.shape[-1] == 4:
            tmp += reference
        else:
            assert reference.shape[-1] == 2
            tmp[..., :4] += reference[...,0:1]
            tmp[..., 4:8] += reference[...,1:2]
        output= self.forward_box_head(tmp,tmp)


        return output

    def forward_box_head(self, hs, memory):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        if self.head_type == "CORNER":
            # adjust shape
            enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
            dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
            att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            # run the corner head
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        elif self.head_type == "MLP":
            '''
            sim_matrix = torch.einsum("abn,abm->anm", hs,
                                      memory)/ self.temperature
            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
            '''
            # Forward the class and box head
            #hs=hs.transpose(1,0).transpose(1,3)  #(B,c,H,w)

            outputs_coord = self.box_head(hs)
            #out = {'pred_boxes': outputs_coord[-1]}
            out=outputs_coord
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_coord)
            return out

    def adjust(self, output_back: list, pos_embed: list):
        """
        """
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}
    ###特征
    def adjust_template(self, output_back, pos_embed,lvl):
        """
        """
        src_feat, mask = output_back.decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck_template[lvl](src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed.flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return  feat_vec, mask_vec, pos_embed_vec
    def adjust_search(self, output_back: list, pos_embed: list,lvl):
        """
        """
        src_feat, mask = output_back.decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck_search[lvl](src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed.flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return feat_vec, mask_vec, pos_embed_vec

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in outputs_coord[:-1]]

    def inverse_sigmoid(self, x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_dual_deform(cfg):
    backbone_template = build_backbone(cfg)# backbone and positional encoding are built together
    backbone_search = build_backbone(cfg)
    transformer = build_deforamble_transformer(cfg)
    box_head = build_box_head(cfg)
    model = match(
        backbone_template,
        backbone_search,
        transformer,
        box_head,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE,
        num_feature_levels=cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS
    )

    return model
