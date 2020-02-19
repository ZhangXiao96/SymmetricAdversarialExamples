from lib.ModelWrapper import ModelWrapper
from tensorboardX import SummaryWriter
import torch
from torchvision import transforms, datasets
from archs.Cifar10 import vgg, resnet
import numpy as np
import os


data_name = 'Cifar10'
model_name = 'resnet'

# init recorder
eval_batch_size = 1

dataset = datasets.CIFAR10
train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
eval_transform = transforms.Compose([transforms.ToTensor()])
if model_name == 'vgg16':
    model = vgg.vgg16()
elif model_name == 'resnet':
    model = resnet.ResNet18()
else:
    raise Exception("No such model!")

# load data
test_data = dataset('D:/Datasets', train=False, transform=eval_transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=0,
                                          drop_last=False)

# build model
save_path = os.path.join('../runs', data_name, model_name)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = model.to(device)
trained_weights = torch.load(os.path.join(save_path, 'ckpt.pkl'))['net']
model.load_state_dict(trained_weights)

from lib.FindFlaws_simple import FindFlaws_simple
wrapper = FindFlaws_simple(model, device)

import matplotlib.pyplot as plt
for x, y in test_loader:
    delta, adv_t1, adv_t2 = wrapper.pert(x, y, 1e-1, 0.01, 100)
    delta = delta.cpu().data.numpy()
    delta = np.squeeze(delta).transpose(1, 2, 0)
    img = x.cpu().data.numpy()
    img = np.squeeze(img).transpose(1, 2, 0)
    adv1 = img + delta
    adv2 = img - delta
    img1 = np.clip(adv1, 0, 1)
    img2 = np.clip(adv2, 0, 1)

    label1 = model(adv_t1)
    label = model(x.to(device))
    label2 = model(adv_t2)

    _, predicted = label.max(1)
    _, predicted1 = label1.max(1)
    _, predicted2 = label2.max(1)
    y1, y0, y2 = predicted1.item(), predicted.item(), predicted2.item()

    plt.subplot(131)
    plt.imshow(img1)
    plt.title(y1)
    plt.subplot(132)
    plt.imshow(img)
    plt.title(y0)
    plt.subplot(133)
    plt.imshow(img2)
    plt.title(y2)
    plt.show()
