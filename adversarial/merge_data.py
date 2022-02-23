import numpy as np

data_1 = np.load('./ori/images_top1aug_ori_3w.npy')
label_1 = np.load('./ori/labels_top1aug_ori_3w.npy')

data_2 = np.load('./cc5/data_fgsm_1w.npy')
label_2 = np.load('./cc5/label_fgsm_1w.npy')

data_3 = np.load('./cc5/data_pgd_1w.npy')
label_3 = np.load('./cc5/label_pgd_1w.npy')

data_4 = np.load('./odi/data_odi_1w.npy')
label_4 = np.load('./odi/label_odi_1w.npy')

#
images=np.concatenate((data_1[:20000], data_2, data_3, data_4), axis=0)
labels=np.concatenate((label_1[:20000], label_2, label_3, label_4), axis=0)

print(images.shape, images.dtype, labels.shape, labels.dtype)
np.save('data.npy', images)
np.save('label.npy', labels)