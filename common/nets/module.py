import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from nets.layer import make_conv_layers, make_conv1d_layers, make_deconv_layers, make_linear_layers
from utils.human_models import mano
from utils.transforms import sample_joint_features, soft_argmax_2d, soft_argmax_3d
from main.config import cfg
from nets.crosstransformer import CrossTransformer
from einops import rearrange
from timm.models.vision_transformer import Block

class Transformer(nn.Module):
    def __init__(self, in_chans=512, joint_num=21, depth=4, num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, joint_num, in_chans))
        self.blocks = nn.ModuleList([
            Block(in_chans, num_heads, mlp_ratio, qkv_bias=False, norm_layer=norm_layer)
            for i in range(depth)])
    def forward(self, x):
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return x
