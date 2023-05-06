'''
this file is for training！！
'''
import random
import sys
sys.path.append("/home/yqm/align-space/Encoder-main")
import os
import matplotlib
import torchvision
import torch
import torch.nn as nn
from torchvision import transforms
from models.sa import SpaceAligner
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.images_dataset import ImagesDataset
from criteria import id_loss, w_norm, moco_loss, vgg_loss, style_loss
from criteria.lpips.lpips import LPIPS
from criteria.prior_loss import compute_mvg
from training.ranger import Ranger
import torch.nn.functional as F
from utils import common, train_utils
import matplotlib.pyplot as plt
from config import data_config
matplotlib.use('Agg')
class Coach(nn.Module):
    def configure_optimizers(self):
        if self.opts.train_encoder:
            params=list(self.net.encoder.parameters())
        if self.opts.train_featEncoder:
            assert  self.opts.use_featEncoder,"when training featEncoder, self.opts.use_featEncoder should be True! "
            params+=list(self.net.featEncoder.parameters())
        if self.opts.train_featEncoder and self.opts.train_encoder:
            params += list(self.net.encoder.parameters())
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer
    def configure_logger(self, dir):
        log_dir = os.path.join(dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir)
    def configure_datasets(self):
        if self.opts.dataset_type not in data_config.DATASETS.keys():
            Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
        print(f'Loading dataset for {self.opts.dataset_type}')
        dataset_args = data_config.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
                                      # target_root=dataset_args['train_target_root'],
                                      opts=self.opts, segm_path=self.opts.train_segm,
                                      segm_transform=transforms_dict['transform_segm'],
                                      # target_transform=transforms_dict['transform_gt_train'],
                                      source_transform=transforms_dict['transform_source']
                                      )
        test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
                                     # target_root=dataset_args['test_target_root'],
                                     opts=self.opts, segm_path=self.opts.test_segm,
                                     segm_transform=transforms_dict['transform_segm'],
                                     # target_transform=transforms_dict['transform_test'],
                                     source_transform=transforms_dict['transform_source']
                                     )
        if self.opts.use_wandb:
            self.wb_logger.log_dataset_wandb(train_dataset, dataset_name="Train")
            self.wb_logger.log_dataset_wandb(test_dataset, dataset_name="Test")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset
    def __init__(self, opts):
        super(Coach, self).__init__()
        self.opts = opts
        self.global_step = 0
        self.device = self.opts.device  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
        # self.opts.device = self.device
        if self.opts.use_wandb:
            #用于可视化的
            from utils.wandb_utils import WBLogger
            self.wb_logger = WBLogger(self.opts)
        # Initialize network
        self.net = SpaceAligner(self.opts).to(self.device)
        # Initialize loss
        self.configure_loss()
        # Initialize optimizer
        self.optimizer = self.configure_optimizers()
        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)
        # Initialize logger
        self.logger = self.configure_logger(opts.exp_dir)
        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps
        self.transform=transforms.Compose([transforms.Resize((512,512))])
        self.transform2=transforms.Compose([transforms.Resize((512,256))])
    def configure_loss(self):
        #l1, l_id暂时弃用
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        # if self.opts.vgg_lambda > 0:
        # self.vgg_loss = vgg_loss.VGGLoss().to(self.device)
        if self.opts.style_lambda > 0:
            self.style_loss = style_loss.StyleLoss().to(self.device)
        # if self.opts.l1_lambda > 0:
        #     self.l1_loss = nn.L1Loss()
        # if self.opts.id_lambda > 0:
        #     self.id_loss = id_loss.IDLoss().to(self.device).eval()


    def train(self):
        #最不好修改的就是这个train , 因为打算使用radom产生随机数， 之后通过
        self.net.train()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):          
                self.optimizer.zero_grad()
                x, segm = batch
                x = x.to(self.device).float()            
                #segm_hw = segm.shape[1:]
                #segm = segm.view(-1, 1, *segm_hw)
                segm = segm.to(self.device)
                segm=segm.unsqueeze(1).repeat(1,3,1,1)
                x_sg2_rec,x_sem_rec, z_inversion = self.net.forward(x)
                x=self.transform(x)
                x_sg2_rec=self.transform2(x_sg2_rec)
                #use_featEncoder True or False 表示了不同的训练阶段
                if self.opts.use_featEncoder:
                    loss, loss_dict = self.CaculateLoss(x,x_sem_rec,z_inversion)
                if  not self.opts.use_featEncoder:
                    loss, loss_dict = self.CaculateLoss(x_sg2_rec,x_sem_rec,z_inversion)               
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
                            # self.log_images(y, y_hat, name='images/train')
                        self.parse_and_log_images(x_sg2_rec, x_sem_rec[:,:,:,128:384], title='images/train')
                if self.global_step % self.opts.board_interval == 0:
                        self.print_metrics(loss_dict, prefix='train')
                        self.log_metrics(loss_dict, prefix='train')
                    # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                    self.best_val_loss = val_loss_dict['loss']
                    self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break
                self.global_step += 1
    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
        if self.opts.use_wandb:
            self.wb_logger.log(prefix, metrics_dict, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def set_segm_dict(self, segm):
        # return segm dict     等想好 local的监督怎么添加再用
        segm_dict = {}
        up_cloth_segm = (segm == 1)  # 上衣segm
        out_cloth_segm = (segm == 2)  # 外套segm
        top_segm = up_cloth_segm + out_cloth_segm
        shirt_dress_segm = (segm == 3)  # 短裙segm
        pants_segm = (segm == 5)  # 裤子segm
        shoes_segm = (segm == 7)  # 鞋子、袜子segm
        bottom_segm = shirt_dress_segm + pants_segm + shoes_segm
        dress_segm = (segm == 4)
        head_segm = (segm == 6)
        background_segm = (segm == 0)
        rest_segm = (segm == 8)
        full_body_segm = ~background_segm
        segm_dict['up_cloth_segm'] = up_cloth_segm
        segm_dict['out_cloth_segm'] = out_cloth_segm
        segm_dict['shirt_dress_segm'] = shirt_dress_segm
        segm_dict['dress_segm'] = dress_segm
        segm_dict['pants_segm'] = pants_segm
        segm_dict['head_segm'] = head_segm
        segm_dict['shoes_segm'] = shoes_segm
        segm_dict['rest_segm'] = rest_segm
        segm_dict['full_segm'] = full_body_segm
        return segm_dict

    def parse_and_log_images(self, x,y, title, index=None):
        if index is None:
            path = os.path.join(self.logger.log_dir, title, f'{str(self.global_step).zfill(5)}.jpg')
        else:
            path = os.path.join(self.logger.log_dir, title,
                                f'{str(self.global_step).zfill(5)}_{str(index).zfill(5)}.jpg')
        # here  use  cpu
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchvision.utils.save_image(torch.cat([x.detach().cpu(),y.detach().cpu()]), path,
                                     normalize=True, scale_each=True, range=(-1, 1), nrow=2)

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            x, segm = batch
            with torch.no_grad():
                # x, y, segm = x.to(self.device).float(), y.to(self.device).float(), segm.to(self.device)
                x = x.to(self.device).float()
                segm = segm.unsqueeze(1).repeat(1, 3, 1, 1)
                segm = segm.to(self.opts.device)
                x_sg2_rec,x_sem_rec, z_inversion = self.net.forward(x)
                x=self.transform(x)
                x_sg2_rec=self.transform2(x_sg2_rec)   
                if self.opts.use_featEncoder:
                    loss, loss_dict = self.CaculateLoss(x,x_sem_rec,z_inversion)
                if  not self.opts.use_featEncoder:
                    loss, loss_dict = self.CaculateLoss(x_sg2_rec,x_sem_rec,z_inversion)
            agg_loss_dict.append(loss_dict)
            self.parse_and_log_images(x,x_sem_rec,
                                      title='images/test',
                                      index='{:04d}'.format(batch_idx))
            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch
        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')
        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        decoder_save_name = f'decoder_{self.global_step}.pt'
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)

        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
                if self.opts.use_wandb:
                    self.wb_logger.log_best_model()
            else:
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def CaculateLoss(self, x,x_sem_rec,z_inversion):
        '''
        :param x: origin image
        :param x_sem_rec:  the image rec after  semanticStylegan
        :param x_sg2_rec:  the image rec after Stylegan2
        :return:
        '''
        #
        loss = 0.0
        loss_dict = {}
        #第二阶段使用FeatEncoder 训练， 重构更多的细节
        if  self.opts.use_featEncoder:
            if self.opts.l2_lambda > 0:
                l2_loss = F.mse_loss(x, x_sem_rec)
                loss_dict['l2_loss'] = float(l2_loss)
                loss += self.opts.l2_lambda * l2_loss
            if self.opts.style_lambda > 0:
                style_loss = self.style_loss(x, x_sem_rec)
                loss_dict['style_loss'] = float(style_loss)
                loss += self.opts.style_lambda * style_loss
            if self.opts.lpips_lambda > 0:
                lpips_loss = self.lpips_loss(x, x_sem_rec)
                loss_dict['lpips_loss'] = float(lpips_loss)
                loss += self.opts.lpips_lambda * lpips_loss
        else:
            #x为 x_sg2_rec , 第一阶段
            assert  len(x.shape)==4,'the tensor x shape should be [batch,channels,h,w]'          
            if self.opts.l2_lambda > 0:
                l2_loss = F.mse_loss(x, x_sem_rec[:,:,:,128:384])
                loss_dict['l2_loss'] = float(l2_loss)
                loss += self.opts.l2_lambda * l2_loss          
            if self.opts.style_lambda > 0:
                style_loss = self.style_loss(x, x_sem_rec[:,:,:,128:384])
                loss_dict['style_loss'] = float(style_loss)
                loss += self.opts.style_lambda * style_loss      
            if self.opts.lpips_lambda > 0:
                lpips_loss = self.lpips_loss(x, x_sem_rec[:,:,:,128:384])
                loss_dict['lpips_loss'] = float(lpips_loss)
                loss += self.opts.lpips_lambda * lpips_loss         
            if self.opts.z_lambda > 0:
                prior_loss = compute_mvg(z_inversion)
                loss_dict['prior_loss'] = float(prior_loss)
                loss += self.opts.z_lambda * prior_loss
            # if self.opts.l1_lambda>0:
            #     L1_loss=self.l1_loss(y_hat,y)
            #     loss_dict['l1_loss']=float(L1_loss)
            #     loss+=self.opts.l1_lambda*L1_loss
            # if self.opts.id_lambda>0:
            #       ID_loss,_,_=self.id_loss(y_hat,y,y)
            #     loss_dict['ID_loss']=float(ID_loss)
            #     loss+=self.opts.id_lambda*ID_loss
        #另外这个关于inversion的 标准高斯怎么限制？

        loss_dict['loss']=float(loss)
        return loss, loss_dict

    def prior_loss(self,latents):
        return compute_mvg(latents)


    def __get_save_dict(self):
        save_dict = {
        'state_dict': self.net.state_dict(),
        'opts': vars(self.opts)
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg
        return save_dict




