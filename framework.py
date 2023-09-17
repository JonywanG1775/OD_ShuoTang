from socket import IP_DEFAULT_MULTICAST_LOOP
import torch
import torch.nn as nn
from torch.autograd import Variable as V
from loss import SurfaceLoss,Dice_Loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode=False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        # self.optimizer = torch.optim.SGD(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.Dice_loss = Dice_Loss(idc=[0,1])
        self.boundary_loss = SurfaceLoss(idc=[1])
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, dist_batch,gt_onehot, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.dist = dist_batch
        self.img_id = img_id
        self.gt_onehot = gt_onehot

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)
            self.dist = V(self.dist.cuda(), volatile=volatile)
        
    def optimize(self):
        batch_dist = self.dist
        batch_dist = batch_dist.to(device)
        self.gt_onehot = self.gt_onehot.to(device)
        self.forward()
        self.optimizer.zero_grad()
        #print(self.img.size())
        pred,bl = self.net.forward(self.img)

        # loss_bl = self.boundary_loss(bl,batch_dist)
        self.mask = self.mask.unsqueeze(1)
        loss = self.loss(self.mask, pred)
        loss_0 = self.Dice_loss(self.gt_onehot,bl)

        loss = 0.5 * loss + 0.5 * loss_0
        loss.backward()
        self.optimizer.step()
        return loss.data, pred
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print ('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr


