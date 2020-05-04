import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
    target = one_hot.scatter_(1, labels.unsqueeze(1), 1)
    
    target = Variable(target)
        
    return target

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        ) 

    def forward(self, z, y):
        z = z.view(z.size(0), 100)
        ret = torch.cat([z, y], 1)
        ret = self.model(ret)
        return ret.view(z.size(0), -1, 28, 28)
         
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = x.view(x.size(0), 784)
        ret = torch.cat([x, y], 1)
        ret = self.model(ret)
        return ret

if __name__ == '__main__':
    batch_size = 1

    generator = Generator()
    generator.cuda()
    noise = torch.rand(batch_size, 100).cuda()
    label = torch.randint(10,(batch_size,)).cuda()
    label_oh = make_one_hot(label, 10)
    gen = generator(noise, label_oh)
    print(gen.shape)

    discriminator = Discriminator()
    discriminator.cuda()
    image = torch.rand(batch_size, 1, 28, 28).cuda()
    dis = discriminator(image, label_oh)
    print(dis.shape)