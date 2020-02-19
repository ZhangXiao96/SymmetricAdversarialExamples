from lib.ModelWrapper import ModelWrapper
from tensorboardX import SummaryWriter
import torch
from torchvision import transforms, datasets
from archs.Cifar10 import vgg, resnet
import os


data_name = 'Cifar10'
model_name = 'resnet'

# train
lr = 0.1
train_batch_size = 128
train_epoch = 200
decay = 0.1
decay_epoch = [100, 150]

# init recorder
eval_batch_size = 256

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
train_data = dataset('D:/Datasets', train=True, download=True, transform=train_transform)
test_data = dataset('D:/Datasets', train=False, transform=eval_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=0,
                                           drop_last=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=0,
                                          drop_last=False)

# build model
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
wrapper = ModelWrapper(model, optimizer, criterion, device)

# train the model
save_path = os.path.join('../runs', data_name, model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
writer = SummaryWriter(logdir=os.path.join(save_path, "log"))

itr_index = 1
wrapper.train()

for id_epoch in range(train_epoch):
    # train loop
    if id_epoch in decay_epoch:
        # load a new optimizer so that the momentum is reset.
        lr *= decay
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        wrapper.optimizer = optimizer

    for id_batch, (inputs, targets) in enumerate(train_loader):
        loss, acc, _ = wrapper.train_on_batch(inputs, targets)
        writer.add_scalar("train acc", acc, itr_index)
        writer.add_scalar("train loss", loss, itr_index)
        print("epoch:{}/{}, batch:{}/{}, loss={}, acc={}".format(id_epoch+1, train_epoch, id_batch+1,
                                                                 len(train_loader), loss, acc))
        itr_index += 1

    wrapper.eval()
    test_loss, test_acc = wrapper.eval_all(test_loader)
    print("testing...")
    print("epoch:{}/{}, batch:{}/{}, loss={}, acc={}".format(id_epoch + 1, train_epoch, id_batch + 1,
                                                             len(train_loader), test_loss, test_acc))
    writer.add_scalar("test acc", test_acc, itr_index)
    writer.add_scalar("test loss", test_loss, itr_index)
    state = {
        'net': model.state_dict(),
        'optim': optimizer.state_dict(),
        'acc': test_acc,
        'epoch': id_epoch,
        'itr': itr_index
    }
    torch.save(state, os.path.join(save_path, "ckpt.pkl"))
    writer.flush()
    # return to train state.
    wrapper.train()

writer.close()

# # shutdown
# os.system('shutdown /s /t 0')
