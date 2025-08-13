import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from common.timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from main.config import cfg
from common.EANet import EANet
from data.dataset import MultipleDatasets

import datetime

# dynamic dataset import
for i in range(len(cfg.trainset_3d)):
    exec('from ' + cfg.trainset_3d[i] + ' import ' + cfg.trainset_3d[i])
for i in range(len(cfg.trainset_2d)):
    exec('from ' + cfg.trainset_2d[i] + ' import ' + cfg.trainset_2d[i])
exec('from ' + cfg.testset + ' import ' + cfg.testset)


class Trainer:
    def __init__(self):
        super(Trainer, self).__init__()
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()
        slurm_id = os.getenv('SLURM_JOB_ID', 'local')
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # logger
        log_dir = cfg.log_dir
        self.log_dir = f'{log_dir}_{current_time}_{slurm_id}'
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        log_name = 'training_log.txt'
        self.logger = colorlogger(self.log_dir, log_name=log_name)

    def get_optimizer(self, model):
        total_params = []
        for module in model.module.trainable_modules:
            total_params += list(module.parameters())
        optimizer = torch.optim.Adam(total_params, lr=cfg.lr)
        return optimizer

    def save_model(self, state, epoch):
        # file_path = osp.join(cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))
        file_path = osp.join(self.log_dir,'snapshot.pth.tar')

        # do not save mano layer weights
        dump_key = []
        for k in state['network'].keys():
            if 'mano_layer' in k:
                dump_key.append(k)
        for k in dump_key:
            state['network'].pop(k, None)

        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):

        ckpt_path = cfg.resume_path
        ckpt = torch.load(ckpt_path) 
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'], strict=False)
        #optimizer.load_state_dict(ckpt['optimizer'])

        self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model, optimizer

    def set_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
    
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset3d_loader = []
        for i in range(len(cfg.trainset_3d)):
            trainset3d_loader.append(eval(cfg.trainset_3d[i])(transforms.ToTensor(), "train"))
        trainset2d_loader = []
        for i in range(len(cfg.trainset_2d)):
            trainset2d_loader.append(eval(cfg.trainset_2d[i])(transforms.ToTensor(), "train"))
       
        valid_loader_num = 0
        if len(trainset3d_loader) > 0:
            trainset3d_loader = [MultipleDatasets(trainset3d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset3d_loader = []
        if len(trainset2d_loader) > 0:
            trainset2d_loader = [MultipleDatasets(trainset2d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset2d_loader = []

        if valid_loader_num > 1:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=True)
        else:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=False)

        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus*cfg.train_batch_size, shuffle=True, num_workers=cfg.num_thread, pin_memory=True, drop_last=True)

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = EANet('train')
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

class Tester:
    def __init__(self):
        super(Tester, self).__init__()
        
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()
        self.resume_path = cfg.resume_path
        ## get the model weight absolute directory
        weight_dir = os.path.dirname(self.resume_path)
        self.logger.info('Model weight directory: {}'.format(weight_dir))
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(weight_dir, f'test_{current_time}')
        # logger
        self.logger = colorlogger(self.log_dir, log_name='test.log')

        

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset_loader = eval(cfg.testset)(transforms.ToTensor(), "test")
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus*cfg.test_batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True)
        
        self.testset = testset_loader
        self.batch_generator = batch_generator

    def _make_model(self):
        # model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        # model_path = '/home/rhong5/research_pro/hand_modeling_pro/EANet/weights/snapshot_29.pth.tar'

        assert os.path.exists(self.resume_path), 'Cannot find model at ' + self.resume_path
        self.logger.info('Load checkpoint from {}'.format(self.resume_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = EANet('test')
        model = DataParallel(model).cuda()
        ckpt = torch.load(self.resume_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model
    
    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.testset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, eval_result):
        self.testset.print_eval_result(eval_result)


