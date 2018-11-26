import copy

import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision.models.resnet import resnet50, Bottleneck
import pdb

def make_model(args):
    return MGN(args)

"""
CSCE 625: Added support for 'Alighed Parts' branch
    adding in support for a higher parts count (args.aligned_parts)
    if args.used_aligned_branch is set
"""
class MGN(nn.Module):
    def __init__(self, args):
        super(MGN, self).__init__()
        self.args = args
        num_classes = args.num_classes

        resnet = resnet50(pretrained=True)

        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        
        if args.pool == 'max':
            pool2d = nn.MaxPool2d
        elif args.pool == 'avg':
            pool2d = nn.AvgPool2d
        else:
            raise Exception()

        self.maxpool_zg_p1 = pool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = pool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = pool2d(kernel_size=(8, 8))
        
        self.localization = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, padding = 1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 20, kernel_size=3, padding = 1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(20,20, kernel_size=3, padding = 1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(20,32, kernel_size=3, padding = 1),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True),
            nn.Conv2d(32,32, kernel_size=3, padding = 1),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True),
            nn.Conv2d(32,32, kernel_size=3, padding = 1),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32* 6 * 2, 128),
            nn.ReLU(True),
            nn.Linear(128,128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 2)
        )

        reduction = nn.Sequential(nn.Conv2d(2048, args.feats, 1, bias=False), nn.BatchNorm2d(args.feats), nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        #self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_0 = nn.Linear(args.feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(args.feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(args.feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(args.feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(args.feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(args.feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(args.feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(args.feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

        # CSCE 625: Aligned Parts Branch, initialize n-parts
        self.fc_id_256_a = None
        self.reduction_n = None
        self.N = args.aligned_parts
        if args.use_aligned_branch:

            reduction_n = []
            fc_id_256_a = []

            # Ensure N is non-zero and even
            if (self.N == 0 or self.N % 2 != 0):
                raise Exception()

            self.pa = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
            self.maxpool_zpa = pool2d(kernel_size=(int(24 / self.N), 8))

            # Initialize reduction and fully connected layers
            for i in range(self.N):
                reduction_n.append(copy.deepcopy(reduction))
                fc_id_256_a.append(nn.Linear(args.feats, num_classes))
                self._init_fc(fc_id_256_a[i])
  
            self.reduction_n = nn.ModuleList(reduction_n)
            #self.fc_id_256_a = nn.ModuleList(fc_id_256_a)

        # CSCE 625: Use mutual learning (extact logits)
        if args.mutual_learning:
            self.fc_ml = nn.Linear(2048, num_classes)
            init.normal_(self.fc_ml.weight, std=0.001)
            init.constant_(self.fc_ml.bias, 0)


    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        #nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        #nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)
        
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 6 * 2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    

    def forward(self, x):
        
        
        if self.args.use_stn:
            x = self.stn(x)

        x = self.backone(x)

        # Branch output from Resnet
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        # Global pooling for each branch
        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        # Part-2 pooling and splitting
        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        # Part-3 pooling and splitting
        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]
        
        # Global and local reductions
        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)

        '''
        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
        '''
        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)
        
        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        # CSCE 625: Aligned Parts Branch
        batch_size = l0_p2.shape[0] 
        # ln_pa = torch.Tensor(self.N, batch_size, 751)
        fn_pa = torch.Tensor(self.N, batch_size, self.args.feats)
        fn_pa_t = torch.Tensor(batch_size, self.N, self.args.feats)
        if self.args.use_aligned_branch:
            
            # Create branch and local pooling
            pa = self.pa(x)
            zpa = self.maxpool_zpa(pa)

            # Create array of locally pooled n parts
            for i in range(self.N):

                # Create next split
                zn_pa = zpa[:, :, i:(i+1), :]

                # Apply reduction
                reduction = self.reduction_n[i]
                # fn_pa = reduction(zn_pa).squeeze(dim=3).squeeze(dim=2)
                fn_pa[i] = reduction(zn_pa).squeeze(dim=3).squeeze(dim=2)

                # Add fully connected portion
                # ln_pa[i] = self.fc_id_256_a[i](fn_pa)
            fn_pa_t = fn_pa.transpose(0, 1)

        logits = None
        if hasattr(self, 'fc_ml'):
            # logits = self.fc_ml(l_p1)
            logits = l_p1

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

        return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3, fn_pa_t

        



