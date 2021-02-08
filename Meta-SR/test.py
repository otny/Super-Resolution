import time
import torch
import torch.nn as nn
import math

outH = 110
outW = 110
outC = 3
inC = 64
kernel_size = 3

N = 1   # batch??
r = 1.1
scale = 1.1
inH = 100
inW = 100


def repeat_x(x):
        scale_int = math.ceil(scale)
        N,C,H,W = x.size()
        x = x.view(N,C,H,1,W,1)

        x = torch.cat([x]*scale_int,3)
        x = torch.cat([x]*scale_int,5).permute(0,3,5,1,2,4)

        return x.contiguous().view(-1, C, H, W)

def repeat_weight(weight, scale, inw,inh):
        k = int(math.sqrt(weight.size(0)))
        outw  =inw * scale
        outh = inh * scale
        weight = weight.view(k, k, -1)
        scale_w = (outw+k-1) // k
        scale_h = (outh + k - 1) // k
        weight = torch.cat([weight] * scale_h, 0)
        weight = torch.cat([weight] * scale_w, 1)

        weight = weight[0:outh,0:outw,:]

        return weight


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

def main():
        rgb_range = 255
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)
        x = torch.rand(N, inC, inH, inW)
        local_weight = torch.rand(outH*outW, outC*inC*kernel_size*kernel_size)   ###   (outH*outW, outC*inC*kernel_size*kernel_size)

        #print(d2)
        up_x = repeat_x(x)         ### the output is (N*r*r,inC,inH,inW)
        print('W =', local_weight.shape, '\tx =', up_x.shape)

        cols = nn.functional.unfold(up_x, 3,padding=1)
        print('W =', local_weight.shape, '\tx =', cols.shape, 'unfold()')

        scale_int = math.ceil(scale)
        local_weight = repeat_weight(local_weight,scale_int,x.size(2),x.size(3))
        print('W =', local_weight.shape, '\tx =', cols.shape)


        cols = cols.contiguous().view(cols.size(0)//(scale_int**2),scale_int**2, cols.size(1), cols.size(2), 1).permute(0,1, 3, 4, 2).contiguous()
        print('W =', local_weight.shape, '\tx =', cols.shape, '.view()')

        local_weight = local_weight.contiguous().view(x.size(2),scale_int, x.size(3),scale_int,-1,3).permute(1,3,0,2,4,5).contiguous()
        print('W =', local_weight.shape, '\tx =', cols.shape)
        local_weight = local_weight.contiguous().view(scale_int**2, x.size(2)*x.size(3),-1, 3)
        print('W =', local_weight.shape, '\tx =', cols.shape)

        
        print('\tW =', local_weight.shape, '\tx =', cols.shape)
        out = torch.matmul(cols,local_weight).permute(0,1,4,2,3)
        # print('out.shape =', out.shape)
        out = out.contiguous().view(x.size(0),scale_int,scale_int,3,x.size(2),x.size(3)).permute(0,3,4,1,5,2)
        out = out.contiguous().view(x.size(0),3, scale_int*x.size(2),scale_int*x.size(3))
        print('out.shape =', out.shape)
        out = add_mean(out)
        # print('out.shape =', out.shape)



if __name__ == "__main__":
        main()