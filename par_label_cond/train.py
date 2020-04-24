import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from model import *
from loss import *
from dataloader import *

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default = 4)
    parser.add_argument('-e', '--epoch', type=int, default = 1000)
    parser.add_argument('-b', '--batch-size', type=int, default = 100)
    parser.add_argument('-d', '--display-step', type=int, default = 600)
    opt = parser.parse_args()
    return opt

def train(opt):
    # Init Model
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    discriminator.train()

    # Load Dataset
    train_dataset = MNISTDataset('train')
    train_data_loader = MNISTDataloader('train', opt, train_dataset)

    # test_dataset = MNISTDataset('test')
    # test_data_loader = MNISTDataloader('test', opt, test_dataset)

    # Set Optimizer
    optim_gen = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optim_dis = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    # Set Loss
    loss = Loss().cuda()

    writer = SummaryWriter()

    for epoch in range(opt.epoch):
        for i in range(len(train_data_loader.data_loader)):
            step = epoch * len(train_data_loader.data_loader) + i + 1
            # load dataset only batch_size
            data1, data2 = train_data_loader.next_batch()
            image1, label1 = data1
            image2, label2 = data2
            image = torch.cat([image1, image2], 3)
            image = Variable(image.cuda())
            real_rand = np.random.random(opt.batch_size)
            label1[real_rand<0.5] = 10
            label2[real_rand>=0.5] = 10
            real_label = Variable(torch.LongTensor(torch.stack([label1, label2], 1)).cuda())

            # train generator
            generator.train()
            optim_gen.zero_grad()

            noise = Variable(torch.randn(opt.batch_size, 100).cuda())
            fake_label = Variable(torch.LongTensor(torch.randint(10, (opt.batch_size, 2))).cuda())
            fake_rand = np.random.random(opt.batch_size)
            fake_label[fake_rand<0.5, 0] = 10
            fake_label[fake_rand>=0.5, 1] = 10
            gen = generator(noise, fake_label)
            validity = discriminator(gen, fake_label)
            loss_gen_l = loss(validity[fake_rand<0.5, 0], torch.ones(opt.batch_size)[fake_rand<0.5].cuda())
            loss_gen_r = loss(validity[fake_rand>=0.5, 1], torch.ones(opt.batch_size)[fake_rand>=0.5].cuda())

            loss_gen_l.backward(retain_graph=True)
            loss_gen_r.backward()
            loss_gen = loss_gen_l + loss_gen_r
            optim_gen.step()
            
            # train discriminator
            optim_dis.zero_grad()

            validity_real = discriminator(image, real_label)
            loss_dis_real_l = loss(validity_real[real_rand<0.5, 0], torch.ones(opt.batch_size)[real_rand<0.5].cuda())
            loss_dis_real_r = loss(validity_real[real_rand>=0.5, 1], torch.ones(opt.batch_size)[real_rand>=0.5].cuda())

            validity_fake = discriminator(gen.detach(), fake_label)
            loss_dis_fake_l = loss(validity_fake[fake_rand<0.5, 0], torch.zeros(opt.batch_size)[fake_rand<0.5].cuda())
            loss_dis_fake_r = loss(validity_fake[fake_rand>=0.5, 1], torch.zeros(opt.batch_size)[fake_rand>=0.5].cuda())
            
            loss_dis_l = (loss_dis_real_l + loss_dis_fake_l)/2
            loss_dis_r = (loss_dis_real_r + loss_dis_fake_r)/2
            loss_dis_l.backward(retain_graph=True)
            loss_dis_r.backward()
            loss_dis = loss_dis_l + loss_dis_r
            optim_dis.step()

            writer.add_scalar('loss/gen', loss_gen, step)
            writer.add_scalar('loss/dis', loss_dis, step)
            
            if step % opt.display_step == 0:
                writer.add_images('image', image[0][0], step, dataformats="HW")
                writer.add_images('result', gen[0][0], step, dataformats="HW")

                print('[Epoch {}] G_loss : {:.2} | D_loss : {:.2}'.format(epoch + 1, loss_gen, loss_dis))
                
                generator.eval()
                noise = torch.randn(9, 100).cuda()
                label = torch.LongTensor(torch.randint(10, (9, 2))).cuda()
                sample_images = generator(noise, label)
                grid = make_grid(sample_images, nrow=3, normalize=True)
                writer.add_image('sample_image', grid, step)

                torch.save(generator.state_dict(), 'checkpoint.pt')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
    opt = get_opt()
    train(opt)