import numpy as np


import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch import optim
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torchsummary import summary

class partial(nn.Module):
    def __init__(self,out_channel,in_channel,stride = 1,dilated=1):
        super(partial, self).__init__()
        self.img= nn.Sequential(
            nn.Conv2d(in_channel,out_channel,5,stride = stride,dilation= dilated,padding=2*dilated),
            #nn.BatchNorm2d(out_channel),
            nn.ELU(),
        )
        self.mask = nn.Sequential(
            nn.Conv2d(1,1,5,stride = stride,padding=2),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm2d(out_channel)

    def forward(self,x,mask):
        x = self.img(x)
        mask = self.mask(mask)
        print(x.shape,mask.shape)
        x = torch.matmul(x,mask)
        x = self.norm(x)

        return x,mask

class partial_transpose(nn.Module):
    def __init__(self, out_channel, in_channel):
        super(partial_transpose, self).__init__()
        self.img= nn.Sequential(
            nn.ConvTranspose2d(in_channel,out_channel,2,stride=2),
            #nn.BatchNorm2d(out_channel),
            nn.ELU(),
        )
        self.mask = nn.Sequential(
            nn.ConvTranspose2d(1,1,2,stride=2),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm2d(out_channel)

    def forward(self, x, mask):
        x = self.img(x)
        mask = self.mask(mask)
        x = x * mask
        x = self.norm(x)

        return x, mask


class coarse_model(nn.Module):
    def __init__(self):
        super(coarse_model,self).__init__()
        self.partial_1 = partial(32,3)
        self.partial_2 = partial(64, 32, 2)
        self.partial_3 = partial(64, 64, 1)
        self.partial_4 = partial(128, 64,2)
        self.partial_5 = partial(128, 128,1)
        self.partial_6 = partial(128, 128,1,2)
        self.partial_7 = partial(128, 128, 1, 4)
        self.partial_8 = partial(128, 128, 1, 8)
        self.partial_9 = partial(128, 128, 1, 16)
        self.partial_10 = partial(128, 128, 1, 1)
        self.partial_11 = partial(128, 128, 1, 1)

        # Image /4 resolution

        self.upsample_1 = partial_transpose(128,128)

        # Image /2 resolution

        self.partial_12 = partial(64, 128, 1, 1)
        self.partial_13 = partial(64, 64, 1, 1)

        self.upsample_2 = partial_transpose(64,64)

        # Image  resolution

        self.partial_14 = partial(32, 64, 1, 1)
        self.partial_15 = partial(16, 32, 1, 1)
        self.partial_16 = partial(3, 16, 1, 1)

    def forward(self,x,mask):
        input_img = x
        input_mask = mask
        x, mask = self.partial_1(x,mask)
        x, mask = self.partial_2(x, mask)
        x, mask = self.partial_3(x, mask)
        x, mask = self.partial_4(x, mask)
        x, mask = self.partial_5(x, mask)
        x, mask = self.partial_6(x, mask)
        x, mask = self.partial_7(x, mask)
        x, mask = self.partial_8(x, mask)
        x, mask = self.partial_9(x, mask)
        x, mask = self.partial_10(x, mask)
        x, mask = self.partial_11(x, mask)

        x,mask = self.upsample_1(x,mask)

        x, mask = self.partial_12(x, mask)
        x, mask = self.partial_13(x, mask)

        x,mask = self.upsample_2(x,mask)

        x, mask = self.partial_14(x, mask)
        x, mask = self.partial_15(x, mask)
        x, mask = self.partial_16(x, mask)

        x = torch.clamp(x,-1.,1.)
        x = input_img * input_mask + x*(1-input_mask)

        return x

class context_attention(nn.Module):
    def __init__(self,batch_size,in_channel,img_size):
        self.batch_size = batch_size
        self.in_channel = in_channel
        self.img_size = img_size
        super(context_attention, self).__init__()
        self.cosine_sim = nn.CosineSimilarity(dim = 1)

        self.conv_regular_1 = nn.Sequential(
            nn.Conv2d(self.img_size**2,256,5,padding=2),
            nn.ELU(),
        )
        self.conv_regular_2 = nn.Sequential(
            nn.Conv2d(256, 128, 5, padding=2),
            nn.ELU(),
        )


    def forward(self,x,mask):
        invert_mask = torch.gt(1-mask,0.5).bool()
        feature = torch.masked_select(x,invert_mask)
        feature = feature.view(self.batch_size,self.in_channel,-1)
        feature_list = list(feature.size())
        feature = torch.unsqueeze(feature, -1)
        feature = torch.unsqueeze(feature, -1)
        feature = feature.repeat(1,1,1,self.img_size,self.img_size)

        x = torch.unsqueeze(x,2)
        x = x.repeat(1,1,feature_list[-1],1,1)

        out = self.cosine_sim(feature,x)
        out_shape = list(out.shape)
        out = out * mask
        print(out.shape)

        pad = torch.zeros(out_shape[0],self.img_size**2-out_shape[1],out_shape[2],out_shape[3],device=out.device)
        out = torch.cat((out,pad),dim=1)
        out = self.conv_regular_1(out)
        out = self.conv_regular_2(out)

        return out

class refine_model(nn.Module):
    def __init__(self,batch_size,img_size):
        self.batch_size = batch_size
        self.img_size = img_size
        super(refine_model, self).__init__()
        self.partial_1 = partial(32 ,3)
        self.partial_2 = partial(64, 32, 2)
        self.partial_3 = partial(64, 64, 1)
        self.partial_4 = partial(128, 64,2)
        self.partial_5 = partial(128, 128,1)
        self.partial_6 = partial(128, 128,2)
        self.partial_7 = partial(128, 128, 1)
        self.partial_8 = partial(128, 128, 1)
        self.partial_9 = partial(128, 128, 1)
        self.partial_10 = partial(128, 128, 1)
        self.partial_11 = partial(64, 128, 1)

        self.context = context_attention(batch_size,64,img_size//8)

        self.partial_12 = partial(128, 128, 1)
        self.partial_13 = partial(128, 128, 1)

        self.upsample_1 = partial_transpose(128,128)
        self.partial_14 = partial(128, 128, 1)
        self.partial_15 = partial(128, 128, 1)

        self.upsample_2 = partial_transpose(128,128)
        self.partial_16 = partial(128, 128, 1)
        self.partial_17 = partial(128, 128, 1)

        self.upsample_3 = partial_transpose(128,128)
        self.partial_18 = partial(128, 128, 1)
        self.partial_19 = partial(128, 128, 1)

        self.upsample_4 = partial_transpose(128,128)
        self.partial_20 = partial(128, 128, 1)
        self.partial_21 = partial(128, 128, 1)

        self.upsample_5 = partial_transpose(128,128)
        self.partial_22 = partial(128, 128, 1)
        self.partial_23 = partial(128, 128, 1)

        self.upsample_6 = partial_transpose(128,128)
        self.partial_24 = partial(128, 128, 1)
        self.partial_25 = partial(128, 128, 1)

        self.partial_26 = partial(128, 256, 1)
        self.partial_27 = partial(64, 128, 1)
        self.partial_28 = partial(32, 64, 1)
        self.partial_29 = partial(3, 32, 1)

    def forward(self,x,mask):
        input_img = x
        input_mask = mask
        x, mask = self.partial_1(x,mask)
        x, mask = self.partial_2(x, mask)
        x, mask = self.partial_3(x, mask)
        x, mask = self.partial_4(x, mask)
        x, mask = self.partial_5(x, mask)
        x, mask = self.partial_6(x, mask)
        x, mask = self.partial_7(x, mask)
        x, mask = self.partial_8(x, mask)
        x, mask = self.partial_9(x, mask)
        x, mask = self.partial_10(x, mask)
        x, mask = self.partial_11(x, mask)

        tmp_mask = mask

        x = self.context(x,mask)
        y = x
        x, mask = self.partial_12(x, mask)
        x, mask = self.partial_13(x, mask)

        x,mask = self.upsample_1(x,mask)
        x, mask = self.partial_14(x, mask)
        x, mask = self.partial_15(x, mask)

        x,mask = self.upsample_2(x,mask)
        x, mask = self.partial_16(x, mask)
        x, mask = self.partial_17(x, mask)

        x,mask = self.upsample_3(x,mask)
        x, mask = self.partial_18(x, mask)
        x, mask = self.partial_19(x, mask)

        y, mask = self.upsample_4(y,tmp_mask)
        y, mask = self.partial_20(y, mask)
        y, mask = self.partial_21(y, mask)

        y, mask = self.upsample_5(y,mask)
        y, mask = self.partial_22(y, mask)
        y, mask = self.partial_23(y, mask)

        y, mask = self.upsample_6(y,mask)
        y, mask = self.partial_24(y, mask)
        y, mask = self.partial_25(y, mask)

        x = torch.cat((x,y),dim=1)

        x,mask = self.partial_26(x,mask)
        x, mask = self.partial_27(x, mask)
        x, mask = self.partial_28(x, mask)
        x, mask = self.partial_29(x, mask)

        x = torch.clamp(x, -1., 1.)
        x = input_img * input_mask + x * (1 - input_mask)

        return x

class discrimin_model(nn.Module):
    def __init__(self):
        super(discrimin_model,self).__init__()
        self.img_1= nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(4,64,5,stride=2,padding=2)),
            nn.Tanh(),
        )
        self.img_2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 5,stride=2, padding=2)),
            nn.Tanh(),
        )
        self.img_3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 5,stride=2, padding=2)),
            nn.Tanh(),
        )
        self.img_4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, 5,stride=2, padding=2)),
            nn.Tanh(),
        )
        self.img_5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, 5,stride=2, padding=2)),
            nn.Tanh(),
        )
        self.img_6 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, 5, stride=2, padding=2)),
            nn.Tanh(),
        )
    def forward(self,x,mask):
        x = torch.cat((x,mask),dim=1)
        x = self.img_1(x)
        x = self.img_2(x)
        x = self.img_3(x)
        x = self.img_4(x)
        x = self.img_5(x)
        x = self.img_6(x)
        return x


class partial_gan():
    def __init__(self,
                 batch_size,
                 img_size):
        self.batch_size = batch_size
        self.img_size = img_size
        self.coarse = coarse_model()
        self.refine = refine_model(self.batch_size,self.img_size)
        self.discrimin = discrimin_model()

        # print(self.coarse)
        # print(self.refine)
        # print(self.discrimin)
    def gradient_penalty(self,real,fake):
        alpha = torch.rand(self.batch_size,1)
        target_size = self.target_img[0]*self.target_img[1]*self.target_img[2]
        alpha = alpha.expand(self.batch_size,target_size).view(self.batch_size,1,32,32)
        alpha = alpha.cuda() if self.is_cuda else alpha
        interpolates = alpha * real + (1-alpha) * fake

        interpolates = interpolates.cuda() if self.is_cuda else interpolates

        interpolates = autograd.Variable(interpolates,requires_grad=True)
        val_inter = self.netD(interpolates)

        grad = autograd.grad(val_inter,interpolates, grad_outputs=torch.ones(val_inter.size()).cuda() if self.is_cuda else torch.ones(
                                  val_inter.size()),create_graph=True,retain_graph=True,only_inputs=True)[0]

        grad = grad.view(grad.size(0),-1)

        grad = ((grad.norm(2,dim=1)-1)**2).mean() * 10

        return grad


test = partial_gan(2,256)
test.discrimin.cuda()

summary(test.discrimin,[(3,256,256),(1,256,256)])