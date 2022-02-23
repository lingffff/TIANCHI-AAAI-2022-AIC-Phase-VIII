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

from tqdm import tqdm

#
# hyperparameters
#
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
epsilons = [1.0]
order = ['preactresnet18']
# order = ['wideresnet', 'preactresnet18']
save_dir = 'se5'

# Use CUDA
use_cuda = torch.cuda.is_available()
seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
#设置产生对抗样例子的参数,参数越大，人眼越能辨别与原图的差距
#epsilons = [.05, .1, .15, .2, .35]
# epsilons = [.05, .15, .3]
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

class DeepFool(object):
    def __init__(self, nb_candidate=10, overshoot=0.02, max_iter=50, clip_min=-1, clip_max=1):
        self.nb_candidate = nb_candidate
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.clip_min = clip_min
        self.clip_max = clip_max

    def attack(self, model, x):
        device = x.device

        with torch.no_grad():
            logits = model(x)
        self.nb_classes = logits.size(-1)
        assert self.nb_candidate <= self.nb_classes, 'nb_candidate should not be greater than nb_classes'

        # preds = logits.topk(self.nb_candidate)[0]
        # grads = torch.stack(jacobian(preds, x, self.nb_candidate), dim=1)
        # grads will be the shape [batch_size, nb_candidate, image_size]

        adv_x = x.clone().requires_grad_()

        iteration = 0
        logits = model(adv_x)
        current = logits.argmax(dim=1)
        if current.size() == ():
            current = torch.tensor([current])
        w = torch.squeeze(torch.zeros(x.size()[1:])).to(device)
        r_tot = torch.zeros(x.size()).to(device)
        original = current

        while ((current == original).any and iteration < self.max_iter):
            predictions_val = logits.topk(self.nb_candidate)[0]
            gradients = torch.stack(jacobian(predictions_val, adv_x, self.nb_candidate), dim=1)
            with torch.no_grad():
                for idx in range(x.size(0)):
                    pert = float('inf')
                    if current[idx] != original[idx]:
                        continue
                    for k in range(1, self.nb_candidate):
                        w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
                        f_k = predictions_val[idx, k] - predictions_val[idx, 0]
                        pert_k = (f_k.abs() + 0.00001) / w_k.view(-1).norm()
                        if pert_k < pert:
                            pert = pert_k
                            w = w_k

                    r_i = pert * w / w.view(-1).norm()
                    r_tot[idx, ...] = r_tot[idx, ...] + r_i

            adv_x = torch.clamp(r_tot + x, self.clip_min, self.clip_max).requires_grad_()
            logits = model(adv_x)
            current = logits.argmax(dim=1)
            if current.size() == ():
                current = torch.tensor([current])
            iteration = iteration + 1

        adv_x = torch.clamp((1 + self.overshoot) * r_tot + x, self.clip_min, self.clip_max)
        return adv_x


def jacobian(predictions, x, nb_classes):
    list_derivatives = []

    for class_ind in range(nb_classes):
        outputs = predictions[:, class_ind]
        derivatives, = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs), retain_graph=True)
        list_derivatives.append(derivatives)

    return list_derivatives


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
    deepfool = DeepFool()
    pbar = tqdm(enumerate(trainloader), total=len(trainloader))
    for i, (inputs, soft_labels) in pbar:
        #Call deepfool
        inputs = inputs.cuda()
        perturbed_inputs = deepfool.attack(model, inputs)

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
    np.save(f'{save_dir}/data_df_1w66.npy', images_glob)
    np.save(f'{save_dir}/label_df_1w66.npy', labels_glob)