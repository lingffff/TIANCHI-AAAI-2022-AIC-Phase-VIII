from __future__ import print_function
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from config import args_resnet, args_densenet
from utils import load_model, AverageMeter, accuracy

import torch.nn as nn
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
'''
利用在1w张原始cifar10上训练好的Resnet模型来生成xifar10的对抗样本
代码参考pytorch 官方tutorials:

https://pytorch.org/tutorials/beginner/fgsm_tutorial.html?highlight=generative%20adversarial

'''
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
epsilons = [8.0/255]
images_glob=[]
labels_glob=[]
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform,mode="train"):
        images = np.load('ori_data.npy')
        labels = np.load('ori_label.npy')
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

def margin_loss(logits,y):
    logit_org = logits.gather(1,y.view(-1,1))
    logit_target = logits.gather(1,(logits - torch.eye(10)[y].to(logits.device) * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    loss = torch.sum(loss)
    return loss

def margin_loss_single_cls(logits, y, cls=0):
    logit_org = logits.gather(1, y.view(-1,1))
    logit_target = logits.gather(1, torch.tensor(cls).expand_as(y).cuda().view(-1,1))
    loss = -logit_org + logit_target
    loss = torch.sum(loss)
    return loss

def main():
    #for arch in ['resnet50', 'densenet121']:
    for arch in ['resnet50']:
        if arch == 'resnet50':
            args = args_resnet
        else:
            args = args_densenet
        assert args['epochs'] <= 200
        if args['batch_size'] > 256:
            # force the batch_size to 256, and scaling the lr
            args['optimizer_hyperparameters']['lr'] *= 256/args['batch_size']
            args['batch_size'] = 256
        # Data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_val = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = MyDataset(transform=transform_train,mode="train")
        valset = MyDataset(transform=transform_val,mode="eval")
        trainloader = data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=4)
        valloader = data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)
        # Model
        model = load_model(arch)
        model.load_state_dict(torch.load('ori/resnet50.pth.tar', map_location='cpu')['state_dict'])
        model = model.cuda()
        # Test
        for epsilon in epsilons:
            train_acc,train_accs_adv = test(valloader, model,epsilon=epsilon)
            print("epsilon:{},train_acc:{},train_accs_adv{}".format(epsilon,train_acc,train_accs_adv))
            #break

def pgd_attack_mluti(model, images, labels, eps=8.0/255, alpha=0.1, iters=100) :
    images = images.cuda()
    labels = labels.cuda()
    targets = labels.argmax(dim=1)
    # loss = margin_loss_single_cls
        
    ori_images = images
    images_return = images
    cost_max = 0.0

    for i in range(2):
        for j in range(10):
            images = ori_images + torch.zeros_like(ori_images).uniform_(-eps, eps)
            step_size = alpha
            for k in range(iters):
                if k == iters // 2 or k == iters * 3 // 4:
                    step_size /= 10

                images.requires_grad = True
                outputs = model(images)

                model.zero_grad()
                cost = margin_loss_single_cls(outputs, targets, cls=j).cuda()
                cost_item = cost.item()

                cost.backward()

                adv_images = images + step_size * images.grad.sign()
                eta = torch.clamp(adv_images - ori_images.data, min=-eps, max=eps)
                images = torch.clamp(ori_images.data + eta, min=0, max=1).detach_()

                if abs(cost_item) > abs(cost_max):
                    images_return = images
                    cost_max = cost_item
 
    return images_return


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
        perturbed_inputs = pgd_attack_mluti(model, inputs, soft_labels, eps=epsilon)

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
    np.save('./mt_pgd/data_pgd_1w_xx.npy', images_glob)
    np.save('./mt_pgd/label_pgd_1w_xx.npy', labels_glob)