import os
import os.path as osp
import sys
import numpy as np

class Config:
    vis = False
    debug = False
    ## dataset
    trainset_3d = ['InterHand26M']
    trainset_2d = []
    testset = 'InterHand26M'

    print_freq = 100
    
    ## model setting
    hand_resnet_type = 50

    ## input, output
    input_img_shape = (256, 256)
    input_hm_shape = (64, 64, 64)
    output_hm_shape = (8, 8, 8)
    bbox_3d_size = 0.3
    sigma = 2.5

    ## training config
    lr = 1e-4
    lr_dec_factor = 10
    lr_dec_epoch = [15,17]
    end_epoch = 40
    train_batch_size = 80
    
    ## testing config
    test_batch_size = 8
    contact_thr = 0.005

    ## others
    num_thread = 40
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    log_dir = osp.join(root_dir, 'log')
    # output_dir = osp.join(root_dir, 'output')
    # model_dir = osp.join(output_dir, 'model_dump')
    # vis_dir = osp.join(output_dir, 'vis')
    
    # result_dir = osp.join(output_dir, 'result')
    # human_model_path = osp.join(root_dir, 'common', 'utils', 'human_model_files')
    human_model_path = '/home/rhong5/research_pro/hand_modeling_pro/HandPoseSD/body_models'
    resume_path = None
    resent50_encoder_path = '/scratch/rhong5/weights/temp_training_weights/InterHand26MClassification/2025-08-10-23-49-58-840114_4076230/best_model.pth'
    # resume_path = '/home/rhong5/research_pro/hand_modeling_pro/EANet/output/model_dump/snapshot.pth.tar' ## 
    # resume_path = '/home/rhong5/research_pro/hand_modeling_pro/EANet/output/gesturePretrainedEncoderVersion/snapshot.pth.tar'
    
    use_gesture_logits= True
    
    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))


cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from common.utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
for i in range(len(cfg.trainset_3d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d[i]))
for i in range(len(cfg.trainset_2d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_2d[i]))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
# make_folder(cfg.model_dir)
# make_folder(cfg.vis_dir)
# make_folder(cfg.log_dir)
# make_folder(cfg.result_dir)
