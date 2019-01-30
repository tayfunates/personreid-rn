# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(512, 256)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        return x


class ReidRNSelf(nn.Module):
    def __init__(self):
        super(ReidRNSelf, self).__init__()

        self.initial1by1 = nn.Conv2d(2048, 32, kernel_size=1, stride=1, padding=0)

        ##(number of filters per object+coordinate of object)*2
        # self.g_fc1 = nn.Linear((2048+2)*2, 256)
        self.g_fc1 = nn.Linear((2048) * 2, 512)
        torch.nn.init.xavier_uniform_(self.g_fc1.weight)
        self.bng1 = nn.BatchNorm1d(512)

        self.g_fc2 = nn.Linear(512, 256)
        torch.nn.init.xavier_uniform_(self.g_fc2.weight)
        self.bng2 = nn.BatchNorm1d(256)

        self.g_fc3 = nn.Linear(256, 256)
        torch.nn.init.xavier_uniform_(self.g_fc3.weight)
        self.bng3 = nn.BatchNorm1d(256)

        self.g_fc4 = nn.Linear(256, 256)
        torch.nn.init.xavier_uniform_(self.g_fc4.weight)
        self.bng4 = nn.BatchNorm1d(256)

        self.f_fc1 = nn.Linear(256, 256)
        torch.nn.init.xavier_uniform_(self.f_fc1.weight)
        self.bnf1 = nn.BatchNorm1d(256)

        # prepare coord tensor
        # def cvt_coord(i):
        #     return [(i/5-2)/2., (i%5-2)/2.]
        #
        # self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        # if args.cuda:
        #     self.coord_tensor = self.coord_tensor.cuda()
        # self.coord_tensor = Variable(self.coord_tensor)
        # np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        # for i in range(25):
        #     np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        # self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

        self.fcout = FCOutputModel()

    def forward(self, x):
        #print(x.shape)
        x = x.squeeze(dim=1)
        #x = self.initial1by1(x)
        #print(x.shape)

        glbl = self.initial1by1(x)
        glbl = F.avg_pool2d(glbl, 2)
        glbl = glbl.view(glbl.size(0), -1)

        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        h = x.size()[2]
        w = x.size()[3]
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb, n_channels, h * w).permute(0, 2, 1)

        # add coordinates
        # x_flat = torch.cat([x_flat, self.coord_tensor],2)

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x26+11)
        x_i = x_i.repeat(1, 32, 1, 1)  # (64x25x25x26+11)
        x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x26+11)
        x_j = x_j.repeat(1, 1, 32, 1)  # (64x25x25x26+11)

        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3)  # (64x25x25x2*26+11)

        # reshape for passing through network
        x_ = x_full.view(mb * h * w * h * w, -1)
        x_ = self.g_fc1(x_)
        # x_ = self.bng1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        # x_ = self.bng2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        # x_ = self.bng3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        # x_ = self.bng4(x_)
        x_ = F.relu(x_)

        # reshape again and sum
        x_g = x_.view(mb, h * w * h * w, -1)
        x_g = x_g.sum(1).squeeze()

        """f"""
        x_f = self.f_fc1(x_g)
        # x_f = self.bnf1(x_f)
        x_f = F.relu(x_f)

        x_f = torch.cat([x_f, glbl], 1) # Also add a global representation

        return self.fcout(x_f)


