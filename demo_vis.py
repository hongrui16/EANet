import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import time
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'common'))
from main.config import cfg
from common.EANet import EANet
from common.utils.preprocessing import load_img, process_bbox, augmentation
from common.utils.vis import save_obj, vis_mesh
from common.utils.human_models import mano
import glob
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', type=str, default=29, dest='test_epoch')
    parser.add_argument('--gpu', type=str, default='0', dest='gpu_ids')
    parser.add_argument('--input', type=str, default='example_image1.png', dest='input')
    parser.add_argument("--img_text_file", type=str, default=None, help="path to the text file containing image paths")
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True
model = EANet('test')
model = DataParallel(model).cuda()
model_path = cfg.resume_path
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()

args.img_text_file = '/scratch/rhong5/dataset/InterHand/InterHand2.6M_5fps_batch1/test_SH0727_img_filepaths.txt'

if args.img_text_file is not None:
    with open(args.img_text_file, 'r') as f:
        img_paths = [line.strip() for line in f.readlines()]
elif not args.input is None:
    img_paths = [args.input]


# /scratch/rhong5/dataset/InterHand/InterHand2.6M_5fps_batch1/images/train/Capture1/0001_neutral_rigid/cam400015/image0751.jpg
valid_cnt = 0
temp_dir = './eanet_demo_output'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
for img_path in img_paths:
    img = load_img(img_path)
    gesture_folder = img_path.split('/')[-3]
    if gesture_folder.startswith('00'):
        hand_type = 'right'
    else:
        hand_type = 'left'
        continue
    print(f'Processing {img_path}, cnt: {valid_cnt} ...')
    height, width = img.shape[:2]
    bbox = [0, 0, width, height]
    bbox = process_bbox(bbox, width, height)
    img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, 'test')
    img = transform(img.astype(np.float32))/255.
    img = img.cuda()[None,:,:,:]
    inputs = {'img': img}
    targets = {}
    meta_info = {}
    with torch.no_grad():
        out = model(inputs, targets, meta_info, 'test')
        
    img = (img[0].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)  
    rmano_mesh = out['rmano_mesh_cam'][0].cpu().numpy()
    lmano_mesh = out['lmano_mesh_cam'][0].cpu().numpy()
    rel_trans = out['rel_trans'][0].cpu().numpy()

    new_filename = img_path.split('/')[-4] + '_' + img_path.split('/')[-3] + '_' + img_path.split('/')[-2] + '_' + img_path.split('/')[-1]


    save_obj(rmano_mesh*np.array([1,-1,-1]), mano.face['right'], f'{temp_dir}/{new_filename}_EANET.obj')
    # save_obj((lmano_mesh+rel_trans)*np.array([1,-1,-1]), mano.face['left'], 'demo_left.obj')
    
    cv2.imwrite(f'{temp_dir}/{new_filename}.jpg', img[:,:,::-1])
    
    # vis_mesh(img, rmano_mesh[:, :2], alpha=0.7)
    
    # #save image
    # cv2.imwrite(f'{temp_dir}/{new_filename}_mesh.jpg', img[:,:,::-1])
    valid_cnt += 1
    if valid_cnt > 100:
        break