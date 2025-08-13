import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
import math
import random
from glob import glob
from pycocotools.coco import COCO
from torchvision import transforms

import os, sys
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from main.config import cfg
from common.utils.human_models import mano
from common.utils.preprocessing import load_img, get_bbox, sanitize_bbox, process_bbox, trans_point2d, augmentation, process_db_coord, process_human_model_output, get_iou
from common.utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_transform_3D, transform_joint_to_other_db
from common.utils.vis import vis_keypoints, vis_mesh, save_obj, vis_3d_skeleton

class Jr():
    def __init__(self, J_regressor,
                 device='cpu'):
        self.device = device
        self.process_J_regressor(J_regressor)

    def process_J_regressor(self, J_regressor):
        J_regressor = J_regressor.clone().detach()
        tip_regressor = torch.zeros_like(J_regressor[:5])
        tip_regressor[0, 745] = 1.0
        tip_regressor[1, 317] = 1.0
        tip_regressor[2, 444] = 1.0
        tip_regressor[3, 556] = 1.0
        tip_regressor[4, 673] = 1.0
        J_regressor = torch.cat([J_regressor, tip_regressor], dim=0)
        new_order = [0, 13, 14, 15, 16,
                     1, 2, 3, 17,
                     4, 5, 6, 18,
                     10, 11, 12, 19,
                     7, 8, 9, 20]
        self.J_regressor = J_regressor[new_order].contiguous().to(self.device)

    def __call__(self, v):
        return torch.matmul(self.J_regressor, v)


def adjust_joint_img(joint_img):
    joint_img[:,0] *= cfg.input_img_shape[1] / cfg.output_hm_shape[2]
    joint_img[:,1] *= cfg.input_img_shape[0] / cfg.output_hm_shape[1]
    joint_img[:,2] = ((joint_img[:,2] / cfg.output_hm_shape[0])*2-1) * (cfg.bbox_3d_size/2)
    joint_img_xy1 = np.concatenate((joint_img[:,:2], np.ones_like(joint_img)[:,0:1]),1)
    joint_img[:,:2] = np.dot(bb2img_trans, joint_img_xy1.T).T[:,:2]
    return joint_img

class InterHand26M(torch.utils.data.Dataset):
    def __init__(self, transform = None, data_split = 'test', debug = False, **kwargs):        
        self.load_one_hand = kwargs.get('load_one_hand', False)
        
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_split = data_split
        self.debug = debug
        self.img_path = osp.join('/scratch/rhong5/dataset/InterHand/InterHand2.6M_5fps_batch1/', 'images')
        self.annot_path = osp.join('/scratch/rhong5/dataset/InterHand/InterHand2.6M_5fps_batch1/', 'annotations')

        self.target_img_filepath = []

                
        self.regressorR = Jr(copy.deepcopy(mano.layer['right'].J_regressor))
        self.regressorL = Jr(copy.deepcopy(mano.layer['left'].J_regressor))

        # IH26M joint set
        self.joint_set = {
                        'joint_num': 42,
                        'joints_name': ('R_Thumb_4', 'R_Thumb_3', 'R_Thumb_2', 'R_Thumb_1', 'R_Index_4', 'R_Index_3', 'R_Index_2', 'R_Index_1', 'R_Middle_4', 'R_Middle_3', 'R_Middle_2', 'R_Middle_1', 'R_Ring_4', 'R_Ring_3', 'R_Ring_2', 'R_Ring_1', 'R_Pinky_4', 'R_Pinky_3', 'R_Pinky_2', 'R_Pinky_1', 'R_Wrist', 'L_Thumb_4', 'L_Thumb_3', 'L_Thumb_2', 'L_Thumb_1', 'L_Index_4', 'L_Index_3', 'L_Index_2', 'L_Index_1', 'L_Middle_4', 'L_Middle_3', 'L_Middle_2', 'L_Middle_1', 'L_Ring_4', 'L_Ring_3', 'L_Ring_2', 'L_Ring_1', 'L_Pinky_4', 'L_Pinky_3', 'L_Pinky_2', 'L_Pinky_1', 'L_Wrist'),
                        'flip_pairs': [ (i,i+21) for i in range(21)],
                        # 'regressor': np.load(osp.join('..', 'data', 'InterHand26M', 'J_regressor_ih26m_mano.npy'))
                        }
        self.joint_set['joint_type'] = {'right': np.arange(0,self.joint_set['joint_num']//2), 'left': np.arange(self.joint_set['joint_num']//2,self.joint_set['joint_num'])}
        self.joint_set['root_joint_idx'] = {'right': self.joint_set['joints_name'].index('R_Wrist'), 'left': self.joint_set['joints_name'].index('L_Wrist')}
        # self.datalist = self.load_data()

        
        # cache_dir = '/home/rhong5/research_pro/hand_modeling_pro/HandPoseSD/dataloader/dataset/InterHand26M/debug_mini_data'
        cache_dir = '/scratch/rhong5/dataset/InterHand/InterHand2.6M_5fps_batch1/cache'
        os.makedirs(cache_dir, exist_ok=True)
        cache_data_file = f'{cache_dir}/{data_split}_all.pkl'

        if osp.exists(cache_data_file):
            print(f'Loading data from cache: {cache_data_file}')
            self.datalist = self.load_data_from_cache(cache_data_file)
        else:
            print(f'Loading data from scratch and saving to cache: {cache_data_file}')
            self.datalist = self.load_data()
            with open(cache_data_file, 'wb') as f:
                import pickle
                pickle.dump(self.datalist, f)
        print(f'Loaded {len(self.datalist)} samples from {self.data_split} split.')
        
    def load_data_from_cache(self, cache_file):
        import pickle
        with open(cache_file, 'rb') as f:
            datalist = pickle.load(f)
            return datalist
            ## shuffle datalist
            # random.shuffle(datalist)
        new_datalist = []
        for data in datalist:
            img_path = data['img_path']
            if not osp.exists(img_path) or not img_path in self.target_img_filepath:
                continue
            new_datalist.append(data)
        return new_datalist
        
    def load_data(self):
        # load annotation
        db = COCO(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_data.json'))
        with open(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_camera.json')) as f:
            cameras = json.load(f)
        with open(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_joint_3d.json')) as f:
            joints = json.load(f)
        with open(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_MANO_NeuralAnnot.json')) as f:
            mano_params = json.load(f)
        if self.data_split == 'test':
            rootnet_path = osp.join(self.annot_path, 'rootnet', 'rootnet_interhand2.6m_output_' + self.data_split + '.json')
            rootnet_result = {}
            with open(rootnet_path) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                rootnet_result[str(annot[i]['annot_id'])] = annot[i]
        aid_list = db.anns.keys()
        ## human annot
        # with open('aid_human_annot_' + self.data_split + '.txt') as f:
        #     aid_list = f.readlines()
        #     aid_list = [int(x) for x in aid_list]
        
        datalist = []
        for aid in aid_list:
            ann = db.anns[aid]                
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            img_path = osp.join(self.img_path, self.data_split, img['file_name'])
            
            hand_type = ann['hand_type']
            # if hand_type not in ['right', 'left', 'interacting']:
            if not osp.exists(img_path):
                continue

            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']
            
            
            if self.load_one_hand and hand_type not in ['right', 'left']:
                continue

            # camera parameters
            t, R = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32).reshape(3), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32).reshape(3,3)
            t = -np.dot(R,t.reshape(3,1)).reshape(3) # -Rt -> t
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32).reshape(2), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32).reshape(2)
            cam_param = {'R': R, 't': t, 'focal': focal, 'princpt': princpt}

            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_trunc = np.array(ann['joint_valid'],dtype=np.float32).reshape(-1,1)
            joint_trunc[self.joint_set['joint_type']['right']] *= joint_trunc[self.joint_set['root_joint_idx']['right']]
            joint_trunc[self.joint_set['joint_type']['left']] *= joint_trunc[self.joint_set['root_joint_idx']['left']]
            if np.sum(joint_trunc) == 0:
                continue

            joint_valid = np.array(joints[str(capture_id)][str(frame_idx)]['joint_valid'], dtype=np.float32).reshape(-1,1)
            joint_valid[self.joint_set['joint_type']['right']] *= joint_valid[self.joint_set['root_joint_idx']['right']]
            joint_valid[self.joint_set['joint_type']['left']] *= joint_valid[self.joint_set['root_joint_idx']['left']]
            if np.sum(joint_valid) == 0:
                continue

            # joint coordinates
            joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32).reshape(-1,3)
            joint_cam = world2cam(joint_world, R, t)
            joint_cam[np.tile(joint_valid==0, (1,3))] = 1. # prevent zero division error
            joint_img = cam2pixel(joint_cam, focal, princpt)

            # bbox
            if ann['hand_type'] in ['right', 'left']:
                hand_bbox = get_bbox(joint_img[self.joint_set['joint_type'][hand_type], :2], joint_valid[self.joint_set['joint_type'][hand_type], 0], extend_ratio=1.25)
            else:
                if np.sum(joint_valid[self.joint_set['joint_type']['right']]) !=0:
                    rhand_bbox = get_bbox(joint_img[self.joint_set['joint_type']['right'], :2], joint_valid[self.joint_set['joint_type']['right'], 0], extend_ratio=1.25)
                else:
                    rhand_bbox = get_bbox(joint_img[self.joint_set['joint_type']['left'], :2], joint_valid[self.joint_set['joint_type']['left'], 0], extend_ratio=1.25)
                if np.sum(joint_valid[self.joint_set['joint_type']['left']]) !=0:
                    lhand_bbox = get_bbox(joint_img[self.joint_set['joint_type']['left'], :2], joint_valid[self.joint_set['joint_type']['left'], 0], extend_ratio=1.25)
                else:
                    lhand_bbox = get_bbox(joint_img[self.joint_set['joint_type']['right'], :2], joint_valid[self.joint_set['joint_type']['right'], 0], extend_ratio=1.25)
                rhand_bbox = [rhand_bbox[0], rhand_bbox[1], rhand_bbox[0]+rhand_bbox[2], rhand_bbox[1]+rhand_bbox[3]]
                lhand_bbox = [lhand_bbox[0], lhand_bbox[1], lhand_bbox[0]+lhand_bbox[2], lhand_bbox[1]+lhand_bbox[3]]
                hand_bbox = [min(rhand_bbox[0], lhand_bbox[0]), min(rhand_bbox[1], lhand_bbox[1]), max(rhand_bbox[2], lhand_bbox[2]), max(rhand_bbox[3],lhand_bbox[3])]
                hand_bbox = [hand_bbox[0], hand_bbox[1], hand_bbox[2]-rhand_bbox[0], hand_bbox[3]-rhand_bbox[1]]

            bbox = process_bbox(hand_bbox, img_width, img_height)
            
            if self.data_split == 'test':
                bbox = np.array(rootnet_result[str(aid)]['bbox'], dtype=np.float32)
                abs_depth = {'right': rootnet_result[str(aid)]['abs_depth'][0], 'left': rootnet_result[str(aid)]['abs_depth'][1]}

            if bbox is None:
                continue

            # mano parameters
            try:
                mano_param = mano_params[str(capture_id)][str(frame_idx)]
            except KeyError:
                mano_param = {'right': None, 'left': None}
            if self.data_split == 'test':
                datalist.append({
                'img_path': img_path,
                'img_shape': (img_height, img_width),
                'bbox': bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
                'joint_trunc': joint_trunc,
                'cam_param': cam_param,
                'mano_param': mano_param,
                'hand_type': hand_type,
                'abs_depth': abs_depth})

            else:
                datalist.append({
                'img_path': img_path,
                'img_shape': (img_height, img_width),
                'bbox': bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
                'joint_trunc': joint_trunc,
                'cam_param': cam_param,
                'mano_param': mano_param,
                'hand_type': hand_type})

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']
        data['cam_param']['t'] /= 1000 # milimeter to meter
        focal = data['cam_param']['focal']
        princpt = data['cam_param']['princpt']
        hand_type = data['hand_type']
            
        if self.debug:
            enforce_flip = True
        else:
            enforce_flip = False
        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split, enforce_flip)
        img = self.transform(img.astype(np.float32))/255.
        
        if self.load_one_hand and hand_type == 'left':
            do_flip = True
            hand_type = 'right' # flip left hand to right hand
        elif self.load_one_hand and hand_type == 'right':
            do_flip = False
        else:
            pass

        # ih26m hand gt
        joint_cam = data['joint_cam'] / 1000. # milimeter to meter
        joint_valid = data['joint_valid']
        rel_trans = joint_cam[self.joint_set['root_joint_idx']['left'],:] - joint_cam[self.joint_set['root_joint_idx']['right'],:]
        rel_trans_valid = joint_valid[self.joint_set['root_joint_idx']['left']] * joint_valid[self.joint_set['root_joint_idx']['right']]
        joint_cam[self.joint_set['joint_type']['right'],:] = joint_cam[self.joint_set['joint_type']['right'],:] - joint_cam[self.joint_set['root_joint_idx']['right'],None,:] # root-relative
        joint_cam[self.joint_set['joint_type']['left'],:] = joint_cam[self.joint_set['joint_type']['left'],:] - joint_cam[self.joint_set['root_joint_idx']['left'],None,:] # root-relative
        joint_img = data['joint_img']
        joint_img = np.concatenate((joint_img[:,:2], joint_cam[:,2:]),1)
        joint_img, joint_cam, joint_valid, joint_trunc, rel_trans = process_db_coord(joint_img, joint_cam, joint_valid, rel_trans, do_flip, img_shape, self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], mano.th_joints_name)

        # mano coordinates (right hand)
        mano_param = data['mano_param']
        if mano_param['right'] is not None:
            mano_param['right']['hand_type'] = 'right'
            rmano_joint_img, rmano_joint_cam, rmano_joint_trunc, rmano_pose, rmano_shape, rmano_mesh_cam = process_human_model_output(mano_param['right'], data['cam_param'], do_flip, img_shape, img2bb_trans, rot)
            rmano_joint_valid = np.ones((mano.sh_joint_num,3), dtype=np.float32)
            rmano_param_valid = np.ones((mano.orig_joint_num*3), dtype=np.float32)
            rmano_shape_valid = np.ones((mano.shape_param_dim), dtype=np.float32)
        else:
            # dummy values
            rmano_joint_img = np.zeros((mano.sh_joint_num,3), dtype=np.float32)
            rmano_joint_cam = np.zeros((mano.sh_joint_num,3), dtype=np.float32)
            rmano_joint_trunc = np.zeros((mano.sh_joint_num,1), dtype=np.float32)
            rmano_pose = np.zeros((mano.orig_joint_num*3), dtype=np.float32)
            rmano_shape = np.zeros((mano.shape_param_dim), dtype=np.float32)
            rmano_joint_valid = np.zeros((mano.sh_joint_num,3), dtype=np.float32)
            rmano_param_valid = np.zeros((mano.orig_joint_num*3), dtype=np.float32)
            rmano_shape_valid = np.zeros((mano.shape_param_dim), dtype=np.float32)
            rmano_mesh_cam = np.zeros((mano.vertex_num,3), dtype=np.float32)

        # mano coordinates (left hand)
        if mano_param['left'] is not None:
            mano_param['left']['hand_type'] = 'left'
            lmano_joint_img, lmano_joint_cam, lmano_joint_trunc, lmano_pose, lmano_shape, lmano_mesh_cam = process_human_model_output(mano_param['left'], data['cam_param'], do_flip, img_shape, img2bb_trans, rot)
            lmano_joint_valid = np.ones((mano.sh_joint_num,3), dtype=np.float32)
            lmano_param_valid = np.ones((mano.orig_joint_num*3), dtype=np.float32)
            lmano_shape_valid = np.ones((mano.shape_param_dim), dtype=np.float32)
        else:
            # dummy values
            lmano_joint_img = np.zeros((mano.sh_joint_num,3), dtype=np.float32)
            lmano_joint_cam = np.zeros((mano.sh_joint_num,3), dtype=np.float32)
            lmano_joint_trunc = np.zeros((mano.sh_joint_num,1), dtype=np.float32)
            lmano_pose = np.zeros((mano.orig_joint_num*3), dtype=np.float32)
            lmano_shape = np.zeros((mano.shape_param_dim), dtype=np.float32)
            lmano_joint_valid = np.zeros((mano.sh_joint_num,3), dtype=np.float32)
            lmano_param_valid = np.zeros((mano.orig_joint_num*3), dtype=np.float32)
            lmano_shape_valid = np.zeros((mano.shape_param_dim), dtype=np.float32)
            lmano_mesh_cam = np.zeros((mano.vertex_num,3), dtype=np.float32)

        if not do_flip:
            mano_joint_img = np.concatenate((rmano_joint_img, lmano_joint_img))
            mano_joint_cam = np.concatenate((rmano_joint_cam, lmano_joint_cam))
            mano_joint_trunc = np.concatenate((rmano_joint_trunc, lmano_joint_trunc))
            mano_pose = np.concatenate((rmano_pose, lmano_pose))
            mano_shape = np.concatenate((rmano_shape, lmano_shape))
            mano_joint_valid = np.concatenate((rmano_joint_valid, lmano_joint_valid))
            mano_param_valid = np.concatenate((rmano_param_valid, lmano_param_valid))
            mano_shape_valid = np.concatenate((rmano_shape_valid, lmano_shape_valid))
            mano_mesh_cam = np.concatenate((rmano_mesh_cam, lmano_mesh_cam))
        else:
            mano_joint_img = np.concatenate((lmano_joint_img, rmano_joint_img))
            mano_joint_cam = np.concatenate((lmano_joint_cam, rmano_joint_cam))
            mano_joint_trunc = np.concatenate((lmano_joint_trunc, rmano_joint_trunc))
            mano_pose = np.concatenate((lmano_pose, rmano_pose))
            mano_shape = np.concatenate((lmano_shape, rmano_shape))
            mano_joint_valid = np.concatenate((lmano_joint_valid, rmano_joint_valid))
            mano_param_valid = np.concatenate((lmano_param_valid, rmano_param_valid))
            mano_shape_valid = np.concatenate((lmano_shape_valid, rmano_shape_valid))
            mano_mesh_cam = np.concatenate((lmano_mesh_cam, rmano_mesh_cam))
        inputs = {'img': img}
        targets = {'joint_img': joint_img, 'mano_joint_img': mano_joint_img, 
                   'joint_cam': joint_cam, 'mano_joint_cam': mano_joint_cam, 
                   'mano_mesh_cam': mano_mesh_cam, 'rel_trans': rel_trans, 
                   'mano_pose': mano_pose, 'mano_shape': mano_shape}
        
        meta_info = {'do_flip': do_flip, 'bb2img_trans': bb2img_trans, 
                     'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 
                     'mano_joint_trunc': mano_joint_trunc, 'mano_joint_valid': mano_joint_valid, 
                     'rel_trans_valid': rel_trans_valid, 'mano_param_valid': mano_param_valid, 
                     'mano_shape_valid': mano_shape_valid, 'is_3D': float(True),
                     'focal': focal, 'princpt': princpt,
                     'img_shape': img_shape, 'img_path': img_path,
                     'hand_type': hand_type,
                     }
        return inputs, targets, meta_info
    
    def lift_joint(self, joint_img, bb2img_trans, root_depth, cam_param, img_width, do_flip=False):
        joint_img[:,0] *= cfg.input_img_shape[1] / cfg.output_hm_shape[2]
        joint_img[:,1] *= cfg.input_img_shape[0] / cfg.output_hm_shape[1]
        joint_img[:,2] = ((joint_img[:,2] / cfg.output_hm_shape[0])*2-1) * (cfg.bbox_3d_size/2)
        joint_img_xy1 = np.concatenate((joint_img[:,:2], np.ones_like(joint_img)[:,0:1]),1)
        joint_img[:,:2] = np.dot(bb2img_trans, joint_img_xy1.T).T[:,:2]
        joint_img[:,2] += root_depth
        if do_flip:
            joint_img[:,0] = img_width - joint_img[:,0] - 1
        joint_cam = pixel2cam(joint_img, cam_param['focal'], cam_param['princpt'])
        return joint_cam

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {
                    'mpjpe_sh': [[None for _ in range(self.joint_set['joint_num'])] for _ in range(sample_num)],
                    'mpjpe_ih': [[None for _ in range(self.joint_set['joint_num'])] for _ in range(sample_num)],
                    'mpvpe_sh': [None for _ in range(sample_num)],
                    'mpvpe_ih': [None for _ in range(sample_num)],
                    }

        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            joint_gt = annot['joint_cam']
            joint_valid = annot['joint_trunc'].reshape(-1)
            focal = annot['cam_param']['focal']
            princpt = annot['cam_param']['princpt']
            out = outs[n]
            
            ## Intag style (joint alignment & scale alignment)
            vert_pred_r = torch.from_numpy(out['rmano_mesh_cam']).cuda()*1000
            vert_pred_l = torch.from_numpy(out['lmano_mesh_cam']).cuda()*1000
            vert_gt_r = torch.from_numpy(out['mano_mesh_cam_target'][:778]).cuda()*1000
            vert_gt_l = torch.from_numpy(out['mano_mesh_cam_target'][-778:]).cuda()*1000
            pred_joint_proj_r = self.regressorR(vert_pred_r)
            pred_joint_proj_l = self.regressorL(vert_pred_l)

            target_gt_r = self.regressorR(vert_gt_r)
            target_gt_l = self.regressorL(vert_gt_l)
            root_r = pred_joint_proj_r[9:10,:]
            root_l = pred_joint_proj_l[9:10,:]
            scale_r = torch.linalg.norm(pred_joint_proj_r[9,:]- pred_joint_proj_r[0,:])
            scale_l = torch.linalg.norm(pred_joint_proj_l[9,:]- pred_joint_proj_l[0,:])
            scale_r_gt = torch.linalg.norm(target_gt_r[9,:] - target_gt_r[0,:])
            scale_l_gt = torch.linalg.norm(target_gt_l[9,:]- target_gt_l[0,:])
            root_r_gt = target_gt_r[9:10,:]
            root_l_gt = target_gt_l[9:10,:]

            vert_pred_r = ((vert_pred_r - root_r) * (scale_r_gt / scale_r)).cpu().detach()
            vert_pred_l = ((vert_pred_l - root_l) * (scale_l_gt / scale_l)).cpu().detach()
            vert_gt_r = ((vert_gt_r - root_r_gt)).cpu().detach()
            vert_gt_l = ((vert_gt_l - root_l_gt)).cpu().detach()
            pred_joint_proj_r = ((pred_joint_proj_r - root_r) * (scale_r_gt / scale_r)).cpu().detach()
            pred_joint_proj_l = ((pred_joint_proj_l - root_l) * (scale_l_gt / scale_l)).cpu().detach()
            
            target_gt_r = (target_gt_r - root_r_gt).cpu().detach()
            target_gt_l = (target_gt_l - root_l_gt).cpu().detach()
            pred_joint_proj = np.concatenate((pred_joint_proj_r, pred_joint_proj_l))
            target_gt = np.concatenate((target_gt_r, target_gt_l))
            vert_gt = np.concatenate((vert_gt_r, vert_gt_l))
            vert_pred = np.concatenate((vert_pred_r,vert_pred_l))
            for j in range(self.joint_set['joint_num']):
                if joint_valid[j]:
                    if annot['hand_type'] == 'right' or annot['hand_type'] == 'left':
                        eval_result['mpjpe_sh'][n][j] = np.sqrt(np.sum((pred_joint_proj[j] - target_gt[j])**2))
                    else:
                        eval_result['mpjpe_ih'][n][j] = np.sqrt(np.sum((pred_joint_proj[j] - target_gt[j])**2))
            # mpvpe
            if annot['hand_type'] == 'right' and annot['mano_param']['right'] is not None:
                eval_result['mpvpe_sh'][n] = np.sqrt(np.sum((vert_gt[:mano.vertex_num,:] - vert_pred[:mano.vertex_num,:])**2,1)).mean()
            elif annot['hand_type'] == 'left' and annot['mano_param']['left'] is not None:
                eval_result['mpvpe_sh'][n] = np.sqrt(np.sum((vert_gt[mano.vertex_num:,:] - vert_pred[mano.vertex_num:,:])**2,1)).mean()
            elif annot['hand_type'] == 'interacting' and annot['mano_param']['right'] is not None and annot['mano_param']['left'] is not None:
                eval_result['mpvpe_ih'][n] = (np.sqrt(np.sum((vert_gt[:mano.vertex_num,:] - vert_pred[:mano.vertex_num,:])**2,1)).mean() + \
                                            np.sqrt(np.sum((vert_gt[mano.vertex_num:,:] - vert_pred[mano.vertex_num:,:])**2,1)).mean())/2.
            
        return eval_result

    def print_eval_result(self, eval_result):
        tot_eval_result = {
                'mpjpe_sh': [[] for _ in range(self.joint_set['joint_num'])],
                'mpjpe_ih': [[] for _ in range(self.joint_set['joint_num'])],
                'mpvpe_sh': [],
                'mpvpe_ih': [],
                }

        # mpjpe (average all samples)
        for mpjpe_sh in eval_result['mpjpe_sh']:
            for j in range(self.joint_set['joint_num']):
                if mpjpe_sh[j] is not None:
                    tot_eval_result['mpjpe_sh'][j].append(mpjpe_sh[j])
        tot_eval_result['mpjpe_sh'] = [np.mean(result) for result in tot_eval_result['mpjpe_sh']]
        for mpjpe_ih in eval_result['mpjpe_ih']:
            for j in range(self.joint_set['joint_num']):
                if mpjpe_ih[j] is not None:
                    tot_eval_result['mpjpe_ih'][j].append(mpjpe_ih[j])
        tot_eval_result['mpjpe_ih'] = [np.mean(result) for result in tot_eval_result['mpjpe_ih']]

        # mpvpe (average all samples)
        for mpvpe_sh in eval_result['mpvpe_sh']:
            if mpvpe_sh is not None:
                tot_eval_result['mpvpe_sh'].append(mpvpe_sh)
        for mpvpe_ih in eval_result['mpvpe_ih']:
            if mpvpe_ih is not None:
                tot_eval_result['mpvpe_ih'].append(mpvpe_ih)


        eval_result = tot_eval_result
        print('MPVPE for all hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_sh'] + eval_result['mpvpe_ih'])))
        print('MPVPE for single hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_sh'])))
        print('MPVPE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_ih'])))
        print('MPJPE for all hand sequences: %.2f mm' % (np.mean(eval_result['mpjpe_sh'] + eval_result['mpjpe_ih'])))
        print('MPJPE for single hand sequences: %.2f mm' % (np.mean(eval_result['mpjpe_sh'])))
        print('MPJPE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['mpjpe_ih'])))


if __name__ == '__main__':
    
    
    load_one_hand = True
    InterHand26M_dataset = InterHand26M(debug=True, load_one_hand = load_one_hand)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mano_layer_right = copy.deepcopy(mano.layer['right']).to(device = device)
    mano_layer_left = copy.deepcopy(mano.layer['left']).to(device = device)

    
    cnt = 10
    while cnt > 0:
        rand_idx = random.randint(0, len(InterHand26M_dataset)-1)
        
        inputs, targets, meta_info = InterHand26M_dataset[rand_idx]
        image = inputs['img'].numpy().transpose(1,2,0) * 255
        image = image.astype(np.uint8)
        image = image.copy()
        original_image = image.copy()
        
        joint_img = targets['joint_img']
        mano_joint_img = targets['mano_joint_img']
        gt_joint_cam = targets['joint_cam']
        gt_mano_joint_cam = targets['mano_joint_cam']
        mano_mesh_cam = targets['mano_mesh_cam']
        focal = meta_info['focal']
        princpt = meta_info['princpt']
        do_flip = meta_info['do_flip']
        img_shape = meta_info['img_shape']
        hand_type = meta_info['hand_type']
        mano_pose = targets['mano_pose']
        mano_shape = targets['mano_shape']
        bb2img_trans = meta_info['bb2img_trans']

        rshape = mano_shape[:mano.shape_param_dim]
        lshape = mano_shape[mano.shape_param_dim:]
        rpose = mano_pose[:mano.orig_joint_num*3]
        lpose = mano_pose[mano.orig_joint_num*3:]
        right_root_pose = rpose[:3]
        left_root_pose = lpose[:3]
        right_hand_pose = rpose[3:]
        left_hand_pose = lpose[3:]
        
        rshape = torch.from_numpy(rshape).to(device).float().unsqueeze(0)
        lshape = torch.from_numpy(lshape).to(device).float().unsqueeze(0)
        right_root_pose = torch.from_numpy(right_root_pose).to(device).float().unsqueeze(0)
        left_root_pose = torch.from_numpy(left_root_pose).to(device).float().unsqueeze(0)
        right_hand_pose = torch.from_numpy(right_hand_pose).to(device).float().unsqueeze(0)
        left_hand_pose = torch.from_numpy(left_hand_pose).to(device).float().unsqueeze(0)
        
        batch_size = 1
        # Visualize the results
        h, w = image.shape[:2]
        # print('joint_img', joint_img[:21])
        joint_img = adjust_joint_img(joint_img)
        # print('joint_img', joint_img[:21])
        
        
        # print('mano_joint_img', mano_joint_img[:21])
        mano_joint_img = adjust_joint_img(mano_joint_img)
        # print('mano_joint_img', mano_joint_img[:21])
        
        
        for i in range(len(joint_img)):
            if joint_img[i, 0]> 0 and joint_img[i, 1] > 0:
                cv2.circle(image, (int(joint_img[i, 0]), int(joint_img[i, 1])), 3, (255, 0, 0), -1)
                
        for i in range(len(mano_joint_img)):
            if mano_joint_img[i, 0]> 0 and mano_joint_img[i, 1] > 0:
                cv2.circle(image, (int(mano_joint_img[i, 0]), int(mano_joint_img[i, 1])), 3, (0, 0, 255), -1)
        
        cv2.imwrite(f'{rand_idx}.jpg', image)
        # root_pose: [bs, 3]
        # hand_pose: [bs, 45]
        # shape: [bs, 10]
        if hand_type == 'right':
            output = mano_layer_right(betas=rshape, hand_pose=right_hand_pose, global_orient=right_root_pose)
        else:
            output = mano_layer_left(betas=lshape, hand_pose=left_hand_pose, global_orient=left_root_pose)

        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        joint_cam = torch.bmm(torch.from_numpy(mano.sh_joint_regressor).to(device)[None,:,:].repeat(batch_size,1,1), mesh_cam)
        joint_cam = joint_cam.squeeze().cpu().numpy()
        joint_img = cam2pixel(joint_cam, focal, princpt)
        print('joint_img', joint_img[:21])
        
        
        for i in range(len(joint_img)):
            if joint_img[i, 0]> 0 and joint_img[i, 1] > 0:
                cv2.circle(original_image, (int(joint_img[i, 0]), int(joint_img[i, 1])), 3, (0, 255, 0), -1)
        cv2.imwrite(f'{rand_idx}_2.jpg', original_image)
        cnt -= 1