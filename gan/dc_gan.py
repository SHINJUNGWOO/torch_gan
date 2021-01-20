import numpy as np


import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim

import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self,DIM):
        self.DIM = DIM
        super(Generator,self).__init__()
        self.preprocess =nn.Sequential(
            nn.Linear(self.DIM,4*4*4*self.DIM),
            #nn.BatchNorm2d(4*4*4*self.DIM),
            nn.Tanh(),

        )
        # output == 4*4*4*self.DIM
        # view 4*self.DIM ,4,4
        self.upsample_1 = nn.Sequential(
            nn.ConvTranspose2d(4*self.DIM,2*self.DIM,2,stride=2),
            nn.BatchNorm2d(2*self.DIM),
            nn.Tanh(),
        )
        # output == 2*self.DIM, 8,8
        self.upsample_2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.DIM, self.DIM, 2, stride=2),
            #nn.BatchNorm2d(self.DIM),
            nn.Tanh(),
        )
        # output == self.DIM, 16,16

        self.upsample_3 = nn.Sequential(
            nn.ConvTranspose2d(self.DIM, self.DIM, 2, stride=2),
            nn.BatchNorm2d(self.DIM),
            nn.Tanh(),
        )
        # output == self.DIM, 32,32
        self.postprocess = nn.Sequential(
            nn.Conv2d(self.DIM,1,5,padding=2),
            nn.Tanh(),
        )
        #output == 3,32,32

    def forward(self,input):
        x = self.preprocess(input)
        x = x.view(-1,4*self.DIM,4,4)
        x = self.upsample_1(x)
        x = self.upsample_2(x)
        x = self.upsample_3(x)
        x = self.postprocess(x)
        #shape ==3,32,32
        return x


class Discriminator(nn.Module):
    def __init__(self,DIM):
        self.DIM = DIM
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, self.DIM, 5, 2, padding=2),
            nn.Tanh(),
            nn.Conv2d(self.DIM, 2 * self.DIM, 5, 2, padding=2),
            nn.Tanh(),
            nn.Conv2d(2 * self.DIM, 4 * self.DIM, 5, 2, padding=2),
            nn.Tanh(),
        )
        #3,32,32 => 4*self.DIM,4,4

        self.post_process =nn.Sequential(
            nn.Linear(4*4*4*self.DIM,1),
            nn.Tanh(),
        )

    def forward(self,input):
        x = self.disc(input)
        x = x.view(-1,4*4*4*self.DIM)
        x = self.post_process(x)
        return x


class Gan:
    def __init__(self):
        self.DIM = 128
        self.batch_size = 64
        self.target_img = (1,32,32)
        self.is_cuda = True
        self.netG = Generator(self.DIM)
        self.netD = Discriminator(self.DIM)

        if self.is_cuda:
            self.netG.cuda()
            self.netD.cuda()
        print(self.netG,self.netD)

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

    def compile(self):
        self.optimizer_disc = optim.Adam(self.netD.parameters(),lr = 1e-4,betas=(0.5,0.9))
        self.optimizer_gen = optim.Adam(self.netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

    def train_disc(self,real):
        self.netD.zero_grad()
        real = autograd.Variable(real)
        real = real.cuda() if self.is_cuda else real
        real_val = self.netD(real)
        real_val = real_val.mean()
        real_val.backward(-1 * torch.tensor(1, dtype=torch.float))

        noise = torch.randn(self.batch_size, self.DIM)
        noise = noise.cuda() if self.is_cuda else noise
        fake = self.netG(noise)
        fake = autograd.Variable(fake)
        fake = fake.cuda() if self.is_cuda else fake

        fake_val = self.netD(fake)
        fake_val = fake_val.mean()
        fake_val.backward(torch.tensor(1, dtype=torch.float))

        gp = self.gradient_penalty(real,fake)
        gp.backward()
        self.optimizer_disc.step()
        #print(real_val,fake_val,gp)
        self.d_cost = real_val - fake_val + gp

    def train_gen(self,show= False):
        self.netG.zero_grad()
        noise = torch.randn(self.batch_size, self.DIM)
        noise = noise.cuda() if self.is_cuda else noise
        noise = autograd.Variable(noise)
        fake = self.netG(noise)
        fake_val = self.netD(fake)
        fake_val = fake_val.mean()
        fake_val.backward(-1*torch.tensor(1, dtype=torch.float))
        self.optimizer_gen.step()

        self.g_cost = fake_val
        if show == True:
            fake = fake.detach().cpu().numpy()
            show_fake= np.transpose(fake[0],(1,2,0))
            show_fake = np.squeeze(show_fake)
            plt.imshow(show_fake,cmap='gray')
            plt.show()

    def train(self,data,epochs):
        self.compile()
        real = next(data)[0]
        for epoch in range(epochs):
            #real = next(data)[0]


            for p in self.netD.parameters():  # reset requires_grad
                p.requires_grad = True
            for p in self.netG.parameters():  # reset requires_grad
                p.requires_grad = False
            for _ in range(5):

                self.train_disc(real)

            for p in self.netD.parameters():  # reset requires_grad
                p.requires_grad = False
            for p in self.netG.parameters():  # reset requires_grad
                p.requires_grad = True

            if epoch%100 ==0 :
                self.train_gen(True)
                torch.save(self.netG,"/content/drive/MyDrive/Colab Notebooks/data/celeba_gen.pt")
                torch.save(self.netD,"/content/drive/MyDrive/Colab Notebooks/data/celeba_disc.pt")
            else:
                self.train_gen(False)

            print(epoch,self.d_cost,self.g_cost)




def celeba(batch_size):
    data_path= "./data/celeba"
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose(
            [transforms.Resize(32),
             transforms.CenterCrop(32),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
        )
    )

    return iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True))

celeba_data = celeba(64)
test = Gan()
test.netG = torch.load("/content/drive/MyDrive/Colab Notebooks/data/celeba_gen.pt")
test.netD = torch.load("/content/drive/MyDrive/Colab Notebooks/data/celeba_disc.pt")
test.train(celeba_data,10000)