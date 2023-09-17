from email.mime import image
from re import L
from tracemalloc import start
import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
from utils import class2one_hot,one_hot2dist
from functools import partial
from operator import itemgetter
from torchvision import transforms
import Constants


def dist_map_transform(resolution, K):
    return transforms.Compose([
        gt_transform(K),
        lambda t: t.cpu().numpy(),
        partial(one_hot2dist, resolution=resolution),
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

def gt_transform(K):
    return transforms.Compose([
        lambda img: np.array(img)[...],
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
        partial(class2one_hot, K=K),
        itemgetter(0)  # Then pop the element to go back to img shape
    ])

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask, courseMask = None,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
        if not courseMask is None:
            courseMask = cv2.warpPerspective(courseMask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
    if courseMask is None:
        return image, mask
    else:
        return image, mask, courseMask

def randomHorizontalFlip(image, mask, u=0.5, courseMask = None):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        if not courseMask is None:
            courseMask = cv2.flip(courseMask,1)
    if courseMask is None:
        return image, mask
    else:
        return image,mask,courseMask

def randomVerticleFlip(image, mask, u=0.5,courseMask=None):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
        if not courseMask is None:
            courseMask = cv2.flip(courseMask,0)
    if courseMask is None:
        return image, mask
    else:
        return image,mask,courseMask

def randomRotate90(image, mask, u=0.5,courseMask=None):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)
        if not courseMask is None:
            courseMask = cv2.flip(courseMask,0)
    if courseMask is None:
        return image, mask
    else:
        return image,mask,courseMask

def loader(img_path, mask_path,dist_trans):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (Constants.img_size, Constants.img_size))

    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

    mask = cv2.resize(mask, (Constants.img_size, Constants.img_size))
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)
    ##i dont know why do this step
    #img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
    mask = np.array(mask, np.float32).transpose(2, 0, 1)

    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    mask = np.squeeze(mask)
    dist_map = dist_trans(mask)

    gt = torch.as_tensor(mask.copy()).long().contiguous().unsqueeze(0)
    gt_one_hot = torch.zeros((2,Constants.img_size,Constants.img_size))
    gt_one_hot.scatter_(0, gt, 1)

    return img, mask,dist_map,gt_one_hot

def read_datasets(image_root,gt_root,vessel_path = None):
    images = []
    masks = []
    vesselImgs = []
    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name)
        if  not vessel_path is None:
            vesselImg = os.path.join(vessel_path,image_name)

        if cv2.imread(image_path) is not None:
            images.append(image_path)
            masks.append(label_path)
            if not vessel_path is None:
                vesselImgs.append(vesselImg)
    if vessel_path is None:
        return images, masks
    else:
        return images,masks,vesselImgs

class ImageFolder(data.Dataset):

    def __init__(self,img_path, gt_path):
        self.img_path = img_path
        self.gt_path = gt_path
        self.disttransform = dist_map_transform([1, 1], 2)
        self.images, self.labels = read_datasets(self.img_path, self.gt_path)

    def __getitem__(self, index):

        img, mask, dis_map,gt_onehot = loader(self.images[index], self.labels[index],dist_trans=self.disttransform)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)

        return img, mask, dis_map,gt_onehot

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)

class ImageFolderVessel(data.Dataset):

    def __init__(self,img_path, gt_path,vessel_path):
        self.img_path = img_path
        self.gt_path = gt_path
        self.vessel_path = vessel_path
        self.images, self.labels,self.vesselImgs = read_datasets(self.img_path, self.gt_path,self.vessel_path)

    def __getitem__(self, index):

        img, mask = self.loader(self.images[index], self.labels[index],self.vesselImgs[index])
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
    
        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)

    def loader(self,img_path, mask_path,vessel_path):
        img = cv2.imread(img_path)

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        
        vesselImg = cv2.imread(vessel_path)
        vesselImg = cv2.cvtColor(vesselImg,cv2.COLOR_BGR2GRAY)

        img = randomHueSaturationValue(img,
                                    hue_shift_limit=(-30, 30),
                                    sat_shift_limit=(-5, 5),
                                    val_shift_limit=(-15, 15))

        img, mask,vesselImg = randomShiftScaleRotate(img, mask,courseMask=vesselImg,
                                        shift_limit=(-0.1, 0.1),
                                        scale_limit=(-0.1, 0.1),
                                        aspect_limit=(-0.1, 0.1),
                                        rotate_limit=(-0, 0))
        img, mask, vesselImg = randomHorizontalFlip(img, mask,courseMask=vesselImg)
        img, mask, vesselImg = randomVerticleFlip(img, mask,courseMask=vesselImg)
        # img, mask, vesselImg = randomRotate90(img, mask,courseMask=vesselImg)

        img_big = np.zeros([Constants.img_size,Constants.img_size,3])
        mask_big = np.zeros([Constants.img_size,Constants.img_size])
        h,w,_ = np.shape(img)

        start_x = (Constants.img_size - h) // 2
        end_x = start_x + h
        start_y = (Constants.img_size - w) // 2
        end_y = start_y + w

        img_big[start_x:end_x,start_y:end_y,:] = img
        # img_big[start_x:end_x,start_y:end_y,2] = vesselImg
        mask_big[start_x:end_x,start_y:end_y] = mask
        mask = np.expand_dims(mask_big, axis=2)
        ##i dont know why do this step
        #img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        img = np.array(img_big, np.float32).transpose(2, 0, 1) / 255.0
        mask = np.array(mask, np.float32).transpose(2, 0, 1)

        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = np.squeeze(mask)

        return img, mask