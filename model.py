import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# https://gist.github.com/davidtvs/3b4c559a0a1ef36478034d9425367ceb
class Maxout(nn.Module):
    def __init__(self, output_channels, axis=1):
        super().__init__()
        self.output_channels = output_channels
        self.axis = axis

    def forward(self, x):
        num_channels = x.size(self.axis)
        if num_channels % self.output_channels:
            raise ValueError(
                "number of input channels({}) is not a multiple of output_channels({})".format(
                    num_channels, self.output_channels
                )
            )
        shape = list(x.shape)
        shape[self.axis] = self.output_channels
        shape.insert(self.axis + 1, num_channels // self.output_channels)

        return x.view(shape).max(self.axis + 1)[0]

# https://gist.github.com/jacobkimmel/4ccdc682a45662e514997f724297f39f
def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x C, where N is batch size. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.
    
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    one_hot = torch.Tensor(labels.size(0), C).zero_().cuda()
    target = one_hot.scatter_(1, labels.data, 1)
    
    target = Variable(target)
        
    return target

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer_z = nn.Sequential(
                            nn.Linear(100, 200),
                            nn.ReLU()
        )
        self.layer_y = nn.Sequential(
                            nn.Linear(10, 1000),
                            nn.ReLU()
        )
        self.layer_cb = nn.Sequential(
                                nn.Linear(1200, 1200),
                                nn.ReLU()
        )
        self.sigmoid = nn.Sequential(
                                nn.Linear(1200, 784),
                                nn.Sigmoid()
        )

    def forward(self, z, y):
        z = self.layer_z(z)
        y = self.layer_y(y)
        ret = self.layer_cb(torch.cat([z,y], 1))
        ret = self.sigmoid(ret)
        return ret
         
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.pieces = 5
        self.pieces_cb = 4
        self.units_x = 240
        self.units_y = 50
        self.units_cb = 240
        
        self.maxout_x = Maxout(self.units_x)
        self.maxout_y = Maxout(self.units_y)
        self.maxout_cb = Maxout(self.units_cb)

        self.layer_x = nn.ModuleList([nn.Linear(3 * 784, self.units_x)]* self.pieces)
        self.layer_y = nn.ModuleList([nn.Linear(10, self.units_y)] * self.pieces)
        self.layer_cb = nn.ModuleList([nn.Linear(self.units_x + self.units_y, self.units_cb)] * self.pieces_cb)

        self.layer_final = nn.Sequential(
                            nn.Linear(self.units_cb, 1),
                            nn.Sigmoid()
        )
    def forward(self, x, y):
        x = x.view(x.shape[0], -1)
        x = [self.layer_x[i](x) for i in range(self.pieces)]
        x = torch.cat(x, 1)
        x = self.maxout_x(x)

        y = [self.layer_y[i](y) for i in range(self.pieces)]
        y = torch.cat(y, 1)
        y = self.maxout_y(y)

        cb = torch.cat([x,y], 1)
        cb = [self.layer_cb[i](cb) for i in range(self.pieces_cb)]
        cb = torch.cat(cb, 1)
        cb = self.maxout_cb(cb)

        ret = self.layer_final(cb)
        return ret

if __name__ == '__main__':
    batch_size = 3

    generator = Generator()
    generator.cuda()
    noise = torch.rand(batch_size, 100).cuda()
    label = torch.randint(10,(batch_size, 10)).cuda()
    label = make_one_hot(label, 10)
    gen = generator(noise, label)
    print(gen.shape)

    discriminator = Discriminator()
    discriminator.cuda()
    image = torch.randn(batch_size, 3, 28, 28).cuda()
    dis = discriminator(image, label)
    print(dis.shape)