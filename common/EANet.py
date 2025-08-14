import torch
import torch.nn as nn
import math
import copy

from torch.nn import functional as F
from einops import rearrange
from timm.models.vision_transformer import Block

from common.nets.layer import make_conv_layers, make_conv1d_layers, make_deconv_layers, make_linear_layers
from common.utils.human_models import mano
from common.utils.transforms import sample_joint_features, soft_argmax_2d, soft_argmax_3d
from common.nets.crosstransformer import CrossTransformer

from common.nets.resnet import ResNetBackbone, Resnet50Encoder
from common.nets.module import Transformer
from common.nets.loss import CoordLoss, ParamLoss
from common.utils.human_models import mano
from common.utils.transforms import rot6d_to_axis_angle
from data.InterHand26M.hand_gesture_names import TWO_HAND_GESTURE_CLASS_MAPPING
from common.modules import CrossFromScalars

from main.config import cfg

main_ids = []
global_sub_ids = []
for class_name, class_info in TWO_HAND_GESTURE_CLASS_MAPPING.items():
    main_ids.append(class_info['main_id'])
    global_sub_ids.append(class_info['global_id'])

main_ids = set(main_ids)
global_sub_ids = set(global_sub_ids)
NUM_MAIN_CLASSES = len(main_ids)
NUM_SUB_CLASSES = len(global_sub_ids)

def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight,std=0.01)
            nn.init.constant_(m.bias,0)
    except AttributeError:
        pass



class FuseFormer(nn.Module):
    def __init__(self):
        super(FuseFormer, self).__init__()
        self.FC = nn.Linear(512*2, 512)
        self.pos_embed = nn.Parameter(torch.randn(1, 1+(2*8*8), 512))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.SA_T = nn.ModuleList([
            Block(512, 4, 4.0, qkv_bias=False, norm_layer=nn.LayerNorm)
            for i in range(4)])
        self.FC2 = nn.Linear(512, 512)
        #Decoder
        self.CA_T = CrossTransformer()
        self.FC3 = nn.Linear(512, 512)

    def forward(self, feat1, feat2):
        B, C, H, W = feat1.shape
        feat1 = rearrange(feat1, 'B C H W -> B (H W) C')
        feat2 = rearrange(feat2, 'B C H W -> B (H W) C')
        # joint Token
        token_j = self.FC(torch.cat((feat1, feat2), dim=-1))
        
        # similar token
        token_s = torch.cat((feat1, feat2), dim=1) + self.pos_embed[:,1:]
        cls_token = (self.cls_token + self.pos_embed[:, :1]).expand(B, -1, -1)
        token_s = torch.cat((cls_token, token_s), dim=1)
        for blk in self.SA_T:
            token_s = blk(token_s)
        token_s = self.FC2(token_s)

        output = self.CA_T(token_j, token_s)
        output = self.FC3(output)
        output = rearrange(output, 'B (H W) C -> B C H W', H=H, W=W)
        return output



class EABlock(nn.Module):
    def __init__(self):
        super(EABlock, self).__init__()
        self.conv_l = make_conv_layers([2048, 512], kernel=1, stride=1, padding=0)
        self.conv_r = make_conv_layers([2048, 512], kernel=1, stride=1, padding=0)
        self.Extract = FuseFormer()
        self.Adapt_r = FuseFormer()
        self.Adapt_l = FuseFormer()
        self.conv_l2 = make_conv_layers([512*2, 512*2], kernel=1, stride=1, padding=0)
        self.conv_r2 = make_conv_layers([512*2, 512*2], kernel=1, stride=1, padding=0)

    def forward(self, hand_feat):
        rhand_feat = self.conv_r(hand_feat)
        lhand_feat = self.conv_l(hand_feat)
        inter_feat = self.Extract(rhand_feat, lhand_feat)
        rinter_feat = self.Adapt_r(rhand_feat, inter_feat)
        linter_feat = self.Adapt_l(lhand_feat, inter_feat)
        rhand_feat = self.conv_r2(torch.cat((rhand_feat,rinter_feat),dim=1))
        lhand_feat = self.conv_l2(torch.cat((lhand_feat,linter_feat),dim=1))
        return rhand_feat, lhand_feat



class PositionNet(nn.Module):
    def __init__(self):
        super(PositionNet, self).__init__()
        self.joint_num = mano.sh_joint_num
        self.EABlock = EABlock()
        self.conv_r2 = make_conv_layers([512*2, self.joint_num*cfg.output_hm_shape[2]], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_l2 = make_conv_layers([512*2, self.joint_num*cfg.output_hm_shape[2]], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, hand_feat):
        rhand_feat, lhand_feat = self.EABlock(hand_feat)
        rhand_hm = self.conv_r2(rhand_feat)
        rhand_hm = rhand_hm.view(-1,self.joint_num, cfg.output_hm_shape[2], cfg.output_hm_shape[0], cfg.output_hm_shape[1])
        rhand_coord = soft_argmax_3d(rhand_hm)

        lhand_hm = self.conv_l2(lhand_feat)
        lhand_hm = lhand_hm.view(-1,self.joint_num, cfg.output_hm_shape[2], cfg.output_hm_shape[0], cfg.output_hm_shape[1])
        lhand_coord = soft_argmax_3d(lhand_hm)

        return rhand_coord, lhand_coord, rhand_feat, lhand_feat



class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()
        self.joint_num = mano.sh_joint_num
        self.rconv = make_conv_layers([1024,512], kernel=1, stride=1, padding=0)
        self.lconv = make_conv_layers([1024,512], kernel=1, stride=1, padding=0)
        self.rshape_out = make_linear_layers([1024, mano.shape_param_dim], relu_final=False)
        self.rcam_out = make_linear_layers([1024, 3], relu_final=False)
        self.lshape_out = make_linear_layers([1024, mano.shape_param_dim], relu_final=False)
        self.lcam_out = make_linear_layers([1024, 3], relu_final=False)
        #SJT
        self.Transformer_r = Transformer(in_chans=512, joint_num=21, depth=4, num_heads=4, mlp_ratio=4., norm_layer=nn.LayerNorm)
        self.Transformer_l = Transformer(in_chans=512, joint_num=21, depth=4, num_heads=4, mlp_ratio=4., norm_layer=nn.LayerNorm)
        #relative translation
        self.root_relative = make_linear_layers([2*(1024),512,3], relu_final=False)
        ##
        self.rroot_pose_out = make_linear_layers([self.joint_num*(512+3), 6], relu_final=False)
        self.rpose_out = make_linear_layers([self.joint_num*(512+3), (mano.orig_joint_num-1)*6], relu_final=False) # without root joint
        self.lroot_pose_out = make_linear_layers([self.joint_num*(512+3), 6], relu_final=False)
        self.lpose_out = make_linear_layers([self.joint_num*(512+3), (mano.orig_joint_num-1)*6], relu_final=False) # without root joint
        self.use_gesture_logits = cfg.use_gesture_logits
        if self.use_gesture_logits:
            self.main_cross_scalar = CrossFromScalars()
            self.sub_cross_scalar = CrossFromScalars()


    def forward(self, rhand_feat, lhand_feat, rjoint_img, ljoint_img, main_logits = None, sub_logits = None):
        batch_size = rhand_feat.shape[0]

        # shape and camera parameters
        rshape_param = self.rshape_out(rhand_feat.mean((2,3)))
        rcam_param = self.rcam_out(rhand_feat.mean((2,3)))
        lshape_param = self.lshape_out(lhand_feat.mean((2,3)))
        lcam_param = self.lcam_out(lhand_feat.mean((2,3)))
        rel_trans = self.root_relative(torch.cat((rhand_feat, lhand_feat), dim=1).mean((2,3)))

        # xyz corrdinate feature
        rhand_feat = self.rconv(rhand_feat)
        lhand_feat = self.lconv(lhand_feat)
        rhand_feat = sample_joint_features(rhand_feat, rjoint_img[:,:,:2]) # batch_size, joint_num, feat_dim
        lhand_feat = sample_joint_features(lhand_feat, ljoint_img[:,:,:2]) # batch_size, joint_num, feat_dim

        if self.use_gesture_logits:
            rhand_feat = self.main_cross_scalar(rhand_feat, main_logits)
            lhand_feat = self.sub_cross_scalar(lhand_feat, sub_logits)

        # import pdb; pdb.set_trace()
        rhand_feat = self.Transformer_r(rhand_feat)
        lhand_feat = self.Transformer_l(lhand_feat)

        # Relative Translation
        rhand_feat = torch.cat((rhand_feat, rjoint_img),2).view(batch_size,-1)
        lhand_feat = torch.cat((lhand_feat, ljoint_img),2).view(batch_size,-1)

        rroot_pose = self.rroot_pose_out(rhand_feat)
        rpose_param = self.rpose_out(rhand_feat)
        lroot_pose = self.lroot_pose_out(lhand_feat)
        lpose_param = self.lpose_out(lhand_feat)

        return rroot_pose, rpose_param, rshape_param, rcam_param, lroot_pose, lpose_param, lshape_param, lcam_param, rel_trans


class EANet(nn.Module):
    def __init__(self, mode='train'):
        super(EANet, self).__init__()
        # self.backbone = ResNetBackbone(cfg.hand_resnet_type)

        
        self.use_gesture_logits = cfg.use_gesture_logits
        self.backbone = Resnet50Encoder(cfg.resent50_encoder_path,
                                        use_gesture_logits=self.use_gesture_logits,
                                        num_main_classes=NUM_MAIN_CLASSES,
                                        num_sub_classes=NUM_SUB_CLASSES)

        self.position_net = PositionNet()
        
        self.rotation_net = RotationNet()
        if mode == 'train':
            self.backbone.init_weights()
            self.position_net.apply(init_weights)
            self.rotation_net.apply(init_weights)
        
        self.mano_layer_right = copy.deepcopy(mano.layer['right']).cuda()
        self.mano_layer_left = copy.deepcopy(mano.layer['left']).cuda()
        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()

        self.trainable_modules = [self.backbone, self.position_net, self.rotation_net]
        self.bce = nn.BCELoss()
        
    def forward_rotation_net(self, rhand_feat, lhand_feat, rhand_coord, lhand_coord, 
                             main_logits = None, sub_logits = None):
        rroot_pose_6d, rpose_param_6d, rshape_param, rcam_param,\
            lroot_pose_6d, lpose_param_6d, lshape_param, \
                lcam_param, rel_trans = self.rotation_net(rhand_feat, lhand_feat, rhand_coord, lhand_coord,
                                                          main_logits=main_logits, sub_logits=sub_logits)
        rroot_pose = rot6d_to_axis_angle(rroot_pose_6d).reshape(-1,3)
        rpose_param = rot6d_to_axis_angle(rpose_param_6d.view(-1,6)).reshape(-1,(mano.orig_joint_num-1)*3)
        lroot_pose = rot6d_to_axis_angle(lroot_pose_6d).reshape(-1,3)
        lpose_param = rot6d_to_axis_angle(lpose_param_6d.view(-1,6)).reshape(-1,(mano.orig_joint_num-1)*3)
        return rroot_pose, rpose_param, rshape_param, rcam_param, lroot_pose, lpose_param, lshape_param, lcam_param, rel_trans
        
    def get_coord(self, root_pose, hand_pose, shape, cam_param, hand_type):
        batch_size = root_pose.shape[0]
        if hand_type == 'right':
            output = self.mano_layer_right(betas=shape, hand_pose=hand_pose, global_orient=root_pose)
        else:
            output = self.mano_layer_left(betas=shape, hand_pose=hand_pose, global_orient=root_pose)

        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        joint_cam = torch.bmm(torch.from_numpy(mano.sh_joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)

        # project 3D coordinates to 2D space
        x = joint_cam[:,:,0] * cam_param[:,None,0] + cam_param[:,None,1]
        y = joint_cam[:,:,1] * cam_param[:,None,0] + cam_param[:,None,2]
        joint_proj = torch.stack((x,y),2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:,mano.sh_root_joint_idx,None,:]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam - root_cam
        return joint_proj, joint_cam, mesh_cam, root_cam

    def forward(self, inputs, targets=None, meta_info=None, mode=None):
        # body network
        if self.use_gesture_logits:
            hand_feat, main_logits, sub_logits = self.backbone(inputs['img'])
        else:
            hand_feat = self.backbone(inputs['img'])
            main_logits = None
            sub_logits = None
        rjoint_img, ljoint_img, rhand_feat, lhand_feat = self.position_net(hand_feat)
        ## rjoint_img, ljoint_img: batch_size, joint_num, 3 (x,y,z), x and y are pixel coordinates, z is in the camera coordination
        
        
        rroot_pose, rhand_pose, rshape, rcam_param, \
            lroot_pose, lhand_pose, lshape, lcam_param, \
                rel_trans = self.forward_rotation_net(rhand_feat, lhand_feat,
                                                      rjoint_img.detach(), ljoint_img.detach(),
                                                      main_logits=main_logits, sub_logits=sub_logits
                                                      )

        ## rel_trans is the relative translation of the two hands wrist, in the camera coordinate
        ## rroot_pose, rhand_pose, lroot_pose, lhand_pose are in axis-angle format, each has shape (batch_size, 3) for root joint and (batch_size, 6) for hand joints
        
        # get outputs
    
        ljoint_proj, ljoint_cam, lmesh_cam, lroot_cam = self.get_coord(lroot_pose, lhand_pose, lshape, lcam_param, 'left')
        rjoint_proj, rjoint_cam, rmesh_cam, rroot_cam = self.get_coord(rroot_pose, rhand_pose, rshape, rcam_param, 'right')

        # combine outputs for the loss calculation (follow mano.th_joints_name)
        mano_pose = torch.cat((rroot_pose, rhand_pose, lroot_pose, lhand_pose),1)
        mano_shape = torch.cat((rshape, lshape),1)
        joint_cam = torch.cat((rjoint_cam, ljoint_cam),1) # mano joint coordinates in the camera coordinate
        joint_img = torch.cat((rjoint_img, ljoint_img),1) # 2.5D joint coordinates, x and y are pixel coordinates, z is the depth in the camera coordinate
        joint_proj = torch.cat((rjoint_proj, ljoint_proj),1) # projected 2D joint coordinates, x and y are pixel coordinates
        
        if mode == 'train':
            loss = {}
            loss['rel_trans'] = self.coord_loss(rel_trans[:,None,:], targets['rel_trans'][:,None,:], meta_info['rel_trans_valid'][:,None,:])
            loss['mano_pose'] = self.param_loss(mano_pose, targets['mano_pose'], meta_info['mano_param_valid'])
            loss['mano_shape'] = self.param_loss(mano_shape, targets['mano_shape'], meta_info['mano_shape_valid'])
            loss['joint_cam'] = self.coord_loss(joint_cam, targets['joint_cam'], meta_info['joint_valid'] * meta_info['is_3D'][:,None,None]) * 10
            loss['mano_joint_cam'] = self.coord_loss(joint_cam, targets['mano_joint_cam'], meta_info['mano_joint_valid']) * 10
            loss['joint_img'] = self.coord_loss(joint_img, targets['joint_img'], meta_info['joint_trunc'], meta_info['is_3D'])
            loss['mano_joint_img'] = self.coord_loss(joint_img, targets['mano_joint_img'], meta_info['mano_joint_trunc'])
            loss['joint_proj'] = self.coord_loss(joint_proj, targets['joint_img'][:,:,:2], meta_info['joint_valid'])
            return loss
        else:
            # test output
            out = {}
            out['img'] = inputs['img']
            out['rel_trans'] = rel_trans
            out['rjoint_img'] = rjoint_img
            out['joint_img'] = joint_img
            out['lmano_mesh_cam'] = lmesh_cam
            out['rmano_mesh_cam'] = rmesh_cam
            out['lmano_root_cam'] = lroot_cam
            out['rmano_root_cam'] = rroot_cam
            out['lmano_joint_cam'] = ljoint_cam
            out['rmano_joint_cam'] = rjoint_cam
            out['lmano_root_pose'] = lroot_pose
            out['rmano_root_pose'] = rroot_pose
            out['lmano_hand_pose'] = lhand_pose
            out['rmano_hand_pose'] = rhand_pose
            out['lmano_shape'] = lshape
            out['rmano_shape'] = rshape
            out['lmano_joint'] = ljoint_proj
            out['rmano_joint'] = rjoint_proj
            if 'mano_joint_img' in targets:
                out['mano_joint_img'] = targets['mano_joint_img']
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            if 'mano_mesh_cam' in targets:
                out['mano_mesh_cam_target'] = targets['mano_mesh_cam']
            if 'do_flip' in meta_info:
                out['do_flip'] = meta_info['do_flip']
            return out

