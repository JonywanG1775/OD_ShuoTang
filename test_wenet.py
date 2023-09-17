import torch
from torch.autograd import Variable as V
import cv2
import os
import numpy as np
import Constants
import warnings
warnings.filterwarnings('ignore')
from networks.W_net import W_Net_

BATCHSIZE_PER_CARD = 8
class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
    def test_one_img_from_path(self, path, evalmode=True,vesselPath=None):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path,vesselPath)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path,vesselPath):
        img = cv2.imread(path)
        vesselImg = cv2.imread(vesselPath)
        
        img_big = np.zeros([Constants.img_size,Constants.img_size,3])
        h,w,_ = np.shape(img)

        start_x = (Constants.img_size - h) // 2
        end_x = start_x + h
        start_y = (Constants.img_size - w) // 2
        end_y = start_y + w

        img_big[start_x:end_x,start_y:end_y,:] = img
        img_big[start_x:end_x,start_y:end_y,2] = vesselImg
        img = img_big
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]
        mask2 = mask2[start_x:end_x,start_y:end_y]
        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).cuda())

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_1(self, path,vesselPath):
        img = cv2.imread(path)
        # vesselImg = cv2.imread(vesselPath)
        # vesselImg = cv2.cvtColor(vesselImg,cv2.COLOR_BGR2GRAY)
        
        img_big = np.zeros([Constants.img_size,Constants.img_size,3])
        h,w,_ = np.shape(img)

        start_x = (Constants.img_size - h) // 2
        end_x = start_x + h
        start_y = (Constants.img_size - w) // 2
        end_y = start_y + w

        img_big[start_x:end_x,start_y:end_y,:] = img
        # img_big[start_x:end_x,start_y:end_y,2] = vesselImg
        img = img_big

        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0
        img5 = V(torch.Tensor(img5).cuda())

        mask,_ = self.net.forward(img5)
        mask = mask.squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]
        mask3 = cv2.resize(mask3,[w,h])
        
        print(np.max(mask3),np.min(mask3))
        _,mask3 = cv2.threshold(mask3,4,255,cv2.THRESH_BINARY)

        return mask3

    def load(self, path):
        model = torch.load(path)
        self.net.load_state_dict(model)


def test_w_net_vessel():
    #source = '/media/songcc/data/songcc/Retinal/dataset/REFUGE/Test400'
    #source = '/media/songcc/data/songcc/Retinal/dataset/REFUGE/Train_original/images'
    source = '/media/songcc/data/songcc/Retinal/DRIVE/test/png_images'
    # source = '/media/songcc/data/songcc/Retinal/DRIVE/training/augImage'
    #source = "/media/songcc/data/songcc/Retinal/dataset/MICCAI2021/train_data"
    #source = "/media/songcc/data/songcc/Retinal/dataset/IDRiD-optic/All_images"
    #vesselPath = '/media/songcc/data/songcc/Retinal/DRIVE/test/courseLabel'
    # vesselPath = '/media/songcc/data/songcc/Retinal/DRIVE/test/courseLabel'
    val = os.listdir(source)
    solver = TTAFrame(W_Net_)
    # # solver.load('weights/log01_dink34-DCENET-DRIVE.th')
    solver.load('weights/W-Net-drive-novessel.pth')
    target = 'result/DRIVE-course/'
    if not os.path.exists(target):
        os.mkdir(target)

    threshold = 4.035
    for i, name in enumerate(val):
        # if i%10 == 0:
        #     print(i/10, '    ','%.2f'%(time()-tic))
        image_path = os.path.join(source, name)
        #vesselImgPath = os.path.join(vesselPath, name)
        mask = solver.test_one_img_from_path(image_path,vesselPath=None)
        mask[mask > threshold] = 255
        mask[mask <= threshold] = 0
        mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)
        cv2.imwrite(target + name, mask.astype(np.uint8))

if __name__ == '__main__':
    test_w_net_vessel()
