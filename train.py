import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse

from torch.utils.tensorboard import SummaryWriter

from model import *
from loss import *
from dataloader import *

MAX_EPOCHS = 500
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default = 4)
    parser.add_argument('-e', '--epoch', type=int, default=40)
    parser.add_argument('-b', '--batch-size', type=int, default = 100)
    parser.add_argument('--test-batch-size', type=int, default = 1)
    opt = parser.parse_args()
    return opt

def train(opt):
    generator = Generator().cuda()
    generator.train()
    discriminator = Discriminator().cuda()
    discriminator.train()

    train_dataset = MNISTDataset('train')
    train_data_loader = MNISTDataloader('train', opt, train_dataset)

    test_dataset = MNISTDataset('test')
    test_data_loader = MNISTDataloader('test', opt, test_dataset)

    optim_gen = torch.optim.SGD(generator.parameters(), lr=0.01, momentum=0.7)
    optim_dis = torch.optim.SGD(discriminator.parameters(), lr=0.01, momentum=0.7)
     
    loss = Loss()

    writer = SummaryWriter()

    for epoch in range(MAX_EPOCHS):
        for i in range(len(train_data_loader.data_loader)):
            step = epoch * len(train_data_loader.data_loader) + i + 1

            image, label = train_data_loader.next_batch()
            image = image.cuda()
            label = label.cuda()
            label = make_one_hot(label, 10)

            optim_gen.zero_grad()

            noise = torch.rand(opt.batch_size, 100).cuda()
            gen = generator(noise, label)
            validity = discriminator(gen, label)

            loss_gen = loss(validity, torch.zeros(opt.batch_size,1).cuda())
            loss_gen.backward()
            optim_gen.step()

            optim_dis.zero_grad()

            validity_real = discriminator(image, label)
            loss_dis_real = loss(validity_real, torch.ones(opt.batch_size,1).cuda())

            validity_fake = discriminator(gen.detach(), label)
            loss_dis_fake = loss(validity_fake, torch.zeros(opt.batch_size,1).cuda())

            loss_dis = (loss_dis_real + loss_dis_fake) / 2
            loss_dis.backward()
            optim_dis.step()

            loss_tot = loss_gen + loss_dis
            

            writer.add_images('image', image[0][0], step, dataformats="HW")
            writer.add_images('result', gen[0][0], step, dataformats="HW")
            writer.add_scalar('loss/total', loss_tot, step)
            writer.add_scalar('loss/gen', loss_gen, step)
            writer.add_scalar('loss/dis', loss_dis, step)
            writer.close()
            
        print('[Epoch {}] Total : {:.2} | G_loss : {:.2} | D_loss : {:.2}'.format(epoch, loss_gen+loss_dis, loss_gen, loss_dis))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    opt = get_opt()
    train(opt)