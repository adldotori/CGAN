from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms

class MNISTDataset(data.Dataset):
    def __init__(self):
        super(MNISTDataset, self).__init__()
        self.dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    
    def name(self):
        return "MNISTDataset"

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

class MNISTDataloader(object):
    def __init__(self, opt, dataset):
        super(MNISTDataloader, self).__init__()
        use_cuda = not torch.cuda.is_available()
        kwargs = {'num_workers': opt.num_workers} if use_cuda else {}

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=True, **kwargs)

        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

if __name__ == '__main__':
    print("[+] Test the Dataset")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default = 12)
    parser.add_argument('-n', '--num-workers', type=int,default = 4)
    opt = parser.parse_args()
    dataset = MNISTDataset()
    data_loader = MNISTDataloader(opt, dataset)

    print('[+] Size of the dataset: %05d, dataloader: %04d' \
        % (len(dataset), len(data_loader.data_loader)))

    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()