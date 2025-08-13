import argparse
from main.config import cfg
import torch
from common.base import Trainer
import torch.backends.cudnn as cudnn
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--debug', dest='debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        
        cfg.debug = True
        cfg.train_batch_size = 20
        cfg.test_batch_size = 20
        cfg.end_epoch = 3
        args.gpu_ids = '0'
        
    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"
 
    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    
    # argument parse and create log
    args = parse_args()
    
    
    cfg.set_args(args.gpu_ids, args.continue_train)
    cudnn.benchmark = True

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    best_loss = float('inf')
    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            loss = trainer.model(inputs, targets, meta_info, 'train')
            loss = {k:loss[k].mean() for k in loss}

            # backward
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            if itr % cfg.print_freq == 0:
                screen = [
                    'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                    'lr: %g' % (trainer.get_lr()),
                    'speed: %.2f(%.2fs r%.2f)s/itr' % (
                        trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                    '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                    ]
                screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
                trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
            
            if args.debug and itr > 5:
                break
        
        if best_loss > loss['mano_joint_cam'].item():
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch)
            best_loss = loss['mano_joint_cam'].item()
            
        if args.debug and epoch > 2:
            break
        

if __name__ == "__main__":
    main()
