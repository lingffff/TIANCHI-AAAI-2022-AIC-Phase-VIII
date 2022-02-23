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

from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
epsilon = 8 / 255
images_glob=[]
labels_glob=[]
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode="train"):
        images = np.load('./data/dataOri.npy')
        labels = np.load('./data/labelOri.npy')
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
    logit_target = logits.gather(1,(logits - torch.eye(10)[y].to("cuda") * 9999).argmax(1, keepdim=True))
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
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = MyDataset(transform=transform_train,mode="train")
        valset = MyDataset(transform=transform_val,mode="eval")
        trainloader = data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=4)
        valloader = data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)
        # Model
        model = load_model(arch)
        model.load_state_dict(torch.load('resnet50_base.pth.tar', map_location='cpu')['state_dict'])
        model = model.cuda()
        # Test

        train_acc,train_accs_adv = test(valloader, model,epsilon=epsilon)
        print("epsilon:{},train_acc:{},train_accs_adv{}".format(epsilon,train_acc,train_accs_adv))
        #break

# FGSM算法攻击代码
def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + epsilon*sign_data_grad
    # 添加剪切以维持[0,1]范围，对于标准化的情况不适用，先注释掉
    #perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回被扰动的图像
    return perturbed_image
    
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
    ODI_num_steps = 10
    num_steps = 140
    ODI_step_size =  8/255 
    step_size =  8/255 
    pbar = tqdm(enumerate(trainloader), total=len(trainloader))
    for i, (inputs, soft_labels) in pbar:
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        inputs = Variable(inputs.data, requires_grad=True)
        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        acc = accuracy(outputs, targets)
        accs.update(acc[0].item(), inputs.size(0))

        X_pgd = Variable(inputs.data, requires_grad=True)
        randVector_ = torch.FloatTensor(*model(X_pgd).shape).uniform_(-1.,1.).to(X_pgd.device)
        if True:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(X_pgd.device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
        for i in range(ODI_num_steps + num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
            with torch.enable_grad():
                if i < ODI_num_steps:
                    loss = (model(X_pgd) * randVector_).sum()
                else:
                    loss = margin_loss(model(X_pgd),targets)
            loss.backward()
            if i < ODI_num_steps: 
                eta = ODI_step_size * X_pgd.grad.data.sign()
            else:
                if i == 56 or i == 102:
                    step_size/=10
                eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - inputs.data, -epsilon, epsilon)
            X_pgd = Variable(inputs.data + eta, requires_grad=True)
            # X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        outputs = model(X_pgd)
        acc = accuracy(outputs, targets)
        accs_adv.update(acc[0].item(), inputs.size(0))
        #--
        save_adv_sample(X_pgd, soft_labels)
        #break
    return  accs.avg,accs_adv.avg


if __name__ == '__main__':
    main()
    images_glob = np.array(images_glob)
    labels_glob = np.array(labels_glob)
    print(images_glob.shape,labels_glob.shape)
    #保存生成的对抗样本用于下一步训练
    np.save('./data/data_odi_1w.npy', images_glob)
    np.save('./data/label_odi_1w.npy', labels_glob)