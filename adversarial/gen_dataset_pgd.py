from __future__ import print_function
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from config import args_wideresnet, args_preactresnet18
from utils import load_model, AverageMeter, accuracy

import torch.nn as nn
from tqdm import tqdm

#
# hyperparameters
#
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
epsilons = [16.0/255]
order = ['wideresnet']
save_dir = 'se5'
# 
# Use CUDA
use_cuda = torch.cuda.is_available()
seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
#设置产生对抗样例子的参数,参数越大，人眼越能辨别与原图的差距
#epsilons = [.05, .1, .15, .2, .35]
#epsilons = [.05, .15, .3]
#epsilons = [.15, .3]
images_glob=[]
labels_glob=[]
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform,mode="train", name=1):
        print(f'best_base/data{name}.npy')
        images = np.load(f'best_base/data{name}.npy')
        labels = np.load(f'best_base/label{name}.npy')
        print("{} :".format(mode),images.shape,labels.shape)
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)

def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)

def main():
    count = 1
    #for arch in ['resnet50', 'densenet121']:
    for arch in order:
        print(arch)
        if arch == 'wideresnet':
            args = args_wideresnet
        else:
            args = args_preactresnet18
        assert args['epochs'] <= 200
        if args['batch_size'] > 256:
            # force the batch_size to 256, and scaling the lr
            args['optimizer_hyperparameters']['lr'] *= 256/args['batch_size']
            args['batch_size'] = 256
        # Data
        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])
        transform_val = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # trainset = MyDataset(transform=transform_train,mode="train")
        valset = MyDataset(transform=transform_val,mode="eval", name=count)
        count += 1
        # trainloader = data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=4)
        valloader = data.DataLoader(valset, batch_size=16, shuffle=False, num_workers=4)
        # Model
        model = load_model(arch)
        model.load_state_dict(torch.load(f'best_base/{arch}.pth.tar', map_location='cpu')['state_dict'])
        model = model.cuda()
        # Test
        for epsilon in epsilons:
            train_acc,train_accs_adv = test(valloader, model,epsilon=epsilon)
            print("epsilon:{},train_acc:{},train_accs_adv{}".format(epsilon,train_acc,train_accs_adv))
            #break

def pgd_attack(model, images, labels, eps, alpha=4.0/255, iters=100, random=True) :
    images = images.cuda()
    labels = labels.cuda()
    targets = labels.argmax(dim=1)
    loss = nn.CrossEntropyLoss()
        
    ori_images = images.data
    
    if random:
        # images = images + (torch.rand(images.size(), dtype=images.dtype, device=images.device) - 0.5) * 2 * eps
        # images = torch.clamp(images, min=0, max=1)
        images = images + torch.zeros_like(images).uniform_(-eps, eps)

    step_size = alpha

    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, targets).cuda()
        cost.backward()

        # if i == iters // 2 or i == iters * 3 // 4:
        #     step_size /= 10

        adv_images = images + step_size * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        # images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        images = (ori_images + eta).detach_()
            
    return images


def save_adv_sample(perturbed_inputs,soft_labels):
    '''
    将生成的对抗样本存储起来
    perturbed_inputs:[B,3,32,32]
    targets:[B,10]
    '''
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2023, 0.1994, 0.2010]
    #print(perturbed_inputs.shape)
    for i in range(perturbed_inputs.shape[0]):
        img=perturbed_inputs[i]
        soft_label=soft_labels[i].cpu().numpy()
        #print(img.shape,soft_label)
        img=img.detach().cpu().numpy()
        #print(img.shape)
        img = np.transpose(img, (1, 2, 0))
        img *= np.array(std)*255
        img += np.array(mean)*255
        img = img.astype(np.uint8)
        #cv2.imwrite('demox6.jpg',img)
        images_glob.append(img)
        labels_glob.append(soft_label)
        #break
    #

def test(trainloader, model, epsilon):
    accs = AverageMeter()
    accs_adv = AverageMeter()
    model.eval()
    #print(len(trainloader))
    pbar = tqdm(enumerate(trainloader), total=len(trainloader))
    for i, (inputs, soft_labels) in pbar:
        #Call PGD Attack
        perturbed_inputs = pgd_attack(model, inputs, soft_labels, eps=epsilon)

        targets = soft_labels.argmax(dim=1).cuda()
        outputs = model(perturbed_inputs)
        acc = accuracy(outputs, targets)
        accs_adv.update(acc[0].item(), inputs.size(0))
        #--
        save_adv_sample(perturbed_inputs,soft_labels)
        #break
    return  accs.avg,accs_adv.avg


if __name__ == '__main__':
    main()
    images_glob = np.array(images_glob)
    labels_glob = np.array(labels_glob)
    print(images_glob.shape,labels_glob.shape)
    #保存生成的对抗样本用于下一步训练
    np.save(f'{save_dir}/data_pgd_1w_small.npy', images_glob)
    np.save(f'{save_dir}/label_pgd_1w_small.npy', labels_glob)