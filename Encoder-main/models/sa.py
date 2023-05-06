'''
this is for inversion w+ latent code
这个python文件需要包含很多东西
(1)创建stylegan2
(2) 利用e4e ， 获取中间的inversion
(3) 进行 style inversion 获得中间的 z2
(4) 使用decoder , 重构图片 ， （这里我采取的方法是重构与经过e4e重构之后得到的图片， 并不重构原始的输入图片）
因为需要进行end2end训练，（利用stylegan的权重是512x256 ，因此需要做裁剪）
 '''
import torch
import torch.nn as nn
from .utils import PixelNorm, EqualLinear, ConvLayer, ResBlock, PositionEmbedding
from config.paths_config import model_paths
from models.coaches.coach3 import E4EInversion
from models.feature_modulation import  LocalFeatEncoder
from models import make_model
import  torchvision
import  torchsummary
import dnnlib
import legacy
def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt

class Inversion_block(nn.Module):
    def __init__(self,n_mlp=8,style_dim=512,lr_mlp=0.01,latent_dim=18):
        super(Inversion_block, self).__init__()
        self.input_layer=EqualLinear(latent_dim,1,lr_mul=lr_mlp,activation='fused_lrelu')
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(
                EqualLinear(style_dim, style_dim,
                    lr_mul=lr_mlp, activation='fused_lrelu')
            )
        self.inversion_style = nn.Sequential(*layers)
    def forward(self,x):
        assert  len(x.shape)==3,'tensor should be [batch, 18, 512]'
        #x=torch.transpose(x.clone().detach().requires_grad_(True),dim0=1,dim1=2)
        x=torch.transpose(x,dim0=1,dim1=2)
        x=self.input_layer(x)
        assert  x.shape[2]==1, 'there is something wrong!'
        return self.inversion_style(x.squeeze(2))

class SpaceAligner(nn.Module):

    def set_sg2(self):
        print("Loading decoder weights from pretrained!",model_paths['Stylegan2'])
        with dnnlib.util.open_url(model_paths['Stylegan2']) as f:
            return legacy.load_network_pkl(f)['G_ema'].to(self.opts.device)
            #这个G_ema 有点动量的意思 ，将取几轮weight 的平均值

    def __init__(self,opts):
        super(SpaceAligner,self).__init__()
        # e4e 是直接预训练好的， 但是因为之前得image得尺寸是 512 X 512的， 如果是将 但是有个问题之前的图片是 1024*512的， 有点难啊 ， 这个
        self.opts=opts
        self.latent_encoder=E4EInversion(self.opts.use_wandb)
        self.encoder=Inversion_block()
        #这个是inversion w+ latent code 使用的 ,应该得到 z （512）维度
        self.sg2_decoder=self.set_sg2()
        if self.opts.use_featEncoder:
            # 把这个东西给忘记了 ， 加上 modulation 模块
            self.featEncoder=LocalFeatEncoder()
        self.decoder=self.set_decoder()

    def forward(self,x):
        '''
        :param x: input image
        (1) compute w_plus via pre-trained e4e
        (2) compute rec_image
        (3) compute sg2_rec_image
        (4)  利用modulation模块 重构一些细节
        :return:x_sem_rec: semanticStylegan : batch, 3, 512,512
        x_sg2_rec: stylegan2(stylehuman) : batch,3, 512,256
        这个感觉还是需要测试一下，来看看crop的位置是不是正确的
        '''
        #x_sg_rec=None
        #x_sem_rec=None
        w_plus=self.latent_encoder.get_image_inversion(x[:,:,:,256:768])
        x_sg2_rec = self.sg2_decoder.synthesis(w_plus, noise_mode='const', force_fp32=True)     
        z_inversion=self.encoder(w_plus)      
        styles= self.decoder.style(z_inversion)     
        x_sem_rec, seg, seg_coarse, depths, _ = self.decoder([styles], input_is_latent=True, randomize_noise=False,
                                                  return_all=True)
        if self.opts.use_featEncoder:
            delta_image = (x - x_sem_rec).detach()
            #
            conditions = self.featEncoder(delta_image)
            if conditions is not None:
                x_sem_rec, result_latent = self.decoder([styles],
                                                     input_is_latent=True,
                                                     randomize_noise=False,
                                                     return_latents=True, conditions=conditions)
        #但是两者之间 ，tensor并不是相同的需要 crop一下
        return x_sg2_rec,x_sem_rec,z_inversion

    def load_weights(self):
        # 这个加载权重的函数还是没有想好， 所以关于inversion style 的部分应该怎么使用？
        if self.opts.checkpoint_path is not None:
            print('Loading inversion_style from pretrained')
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
        if self.opts.checkpoint_path is not None:
            print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.featEncoder.load_state_dict(ckpt, strict=False)
            # 要写一下关于featEncoder加载权重
        else:
            print('Loading encoders weights from e4e!')
            #但是这个地方不对的
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            # if input to encoder is not an RGB image, do not load the input layer weights
            # 这个位置的代码该如何进行修改
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.ckpt)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=False)

    def set_decoder(self):
        self.ckpt = torch.load(self.opts.ckpt)
        self.ckpt['args'].num_workers = 1
        self.decoder = make_model(self.ckpt['args'])
        return self.decoder.to(self.opts.device)


