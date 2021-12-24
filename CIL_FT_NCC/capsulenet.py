"""
Pytorch implementation of CapsNet in paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this.

Usage:
       Launch `python CapsNet.py -h` for usage help

Result:
    Validation accuracy > 99.6% after 50 epochs.
    Speed: About 73s/epoch on a single GTX1070 GPU card and 43s/epoch on a GTX1080Ti GPU.

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Pytorch`
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from capsulelayers import DenseCapsule, PrimaryCapsule, ChannelAttention

import numpy as np
import modified_linear

class Permutate(nn.Module):
    def __init__(self, *args):
        super(Permutate, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.permute(self.shape)

class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()

    def forward(self, x):
        return x.norm(dim=-1)


#### ResNet Backbone ####
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.bn2(self.conv2(out))
        out2 = self.shortcut(x)
        out = out1 + out2
        out = F.relu(out)
        return out


class resnet_backbone(nn.Module):
    def __init__(self, cl_input_channels, cl_num_filters, cl_stride):
        super(resnet_backbone, self).__init__()
        self.in_planes = 128

        def _make_layer(block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        self.pre_caps = nn.Sequential(
            nn.Conv2d(in_channels=cl_input_channels,
                      out_channels=128,
                      kernel_size=6,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(128, momentum=0.9, affine=False),
            nn.ReLU(),
            #_make_layer(block=BasicBlock, planes=128, num_blocks=1, stride=0),  # num_blocks=2 or 3
            _make_layer(block=BasicBlock, planes=cl_num_filters, num_blocks=1, stride=cl_stride),  # num_blocks=2 or 4
        )

    def forward(self, x):
        out = self.pre_caps(x)  # x is an image
        return out


#### Capsule Network ####
class CapsuleNet(nn.Module):
    """
    A Capsule Network on MNIST.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """
    def __init__(self, input_size, num_output_capsules, classes, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.num_output_capsules = num_output_capsules
        self.classes = classes
        self.routings = routings


        # base
        self.permute = Permutate(0, 3, 1, 2)

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=6, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9, affine=True)
        self.activation = nn.ReLU()

        # ATTENTION MODULE
        self.channel_attention = ChannelAttention(64, ratio=2)


        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        # aqui el stride = 2 originalmente
        self.primarycaps = PrimaryCapsule(in_channels=64, out_channels=8*16, dim_caps=8, kernel_size=6, stride=2, padding=0)


        # ===> AGREGAR VERSION de PrimaryCapsule (Configuración inicial)

        # Layer 3: Capsule layer. Routing algorithm works here.
        #   DREAMER: 1888
        #   DEAP: 3584
        self.digitcaps = DenseCapsule(in_num_caps=1888, in_dim_caps=8,
                                      out_num_caps=num_output_capsules, out_dim_caps=16, routings=routings)
        self.norm = Norm()
        self.fc = modified_linear.CosineLinear(num_output_capsules, classes)

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(16*num_output_capsules, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid()
        )



    def forward(self, x, y=None):
        # Reorder Channels, Time, Electrodes
        x = self.permute(x)
        #print(x.size())

        # Layer 1: Just a conventional Conv2D layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        #print(x.size())

        # Channel attention
        x = self.channel_attention(x)
        #print(x.size())

        # Layer: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        x = self.primarycaps(x)
        #print(x.size())

        # Layer: digit capsules
        x = self.digitcaps(x)

        # normalize capsules
        length = x.mean(dim=-1)

        # Fully connected
        length = self.fc(length)
        #print(length.size())

        # during testing, no label given. create one-hot coding using `length`
        if y is None:
            index = length.max(dim=1)[1]
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())

        # Encoder
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return length, reconstruction.view(-1, *self.input_size)


def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
    :param x_recon: reconstructed data, size is same as `x`
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """
    #print(y_true.size())
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()

    L_recon = nn.MSELoss()(x_recon, x)

    return L_margin + lam_recon * L_recon


def show_reconstruction(model, test_loader, n_images, args):
    import matplotlib.pyplot as plt
    from utils_capsnet import combine_images
    from PIL import Image
    import numpy as np

    model.eval()
    for x, _ in test_loader:
        x = Variable(x[:min(n_images, x.size(0))].cuda(), volatile=True)
        _, x_recon = model(x)
        data = np.concatenate([x.data, x_recon.data])
        img = combine_images(np.transpose(data, [0, 2, 3, 1]))
        image = img * 255
        Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
        print()
        print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
        print('-' * 70)
        plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png", ))
        plt.show()
        break


def test(model, test_loader, args, num_classes):
    model.eval()
    test_loss = 0
    correct = 0
    for x, y in test_loader:
        # one-hot vector to calculate loss
        y1 = torch.zeros(y.size(0), num_classes).scatter_(1, y.view(-1, 1), 1.)
        y1 = Variable(y1.cuda()) # to GPU

        # one-hot vector to get predictions
        y = torch.zeros(y.size(0), args.num_output_capsules).scatter_(1, y.view(-1, 1), 1.)
        x, y = Variable(x.cuda()), Variable(y.cuda()) # to GPU

        # compute predictions
        y_pred, x_recon = model(x, y)

        # compute loss
        test_loss += caps_loss(y1, y_pred, x, x_recon, args.lam_recon).item() * x.size(0)  #data[0] # sum up batch loss

        y_pred, x_recon = model(x, y)

        # compute accuracy
        y_pred = y_pred.data.max(1)[1]
        y_true = y1.data.max(1)[1]
        correct += y_pred.eq(y_true).cpu().sum()


    test_loss /= len(test_loader.dataset)

    return test_loss, correct.cpu().numpy() / float(len(test_loader.dataset))


def train(model, train_loader, test_loader, args, num_classes):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param train_loader: torch.utils.data.DataLoader for training data
    :param test_loader: torch.utils.data.DataLoader for test data
    :param args: arguments
    :return: The trained model
    """
    print('Begin Training' + '-'*70)
    from time import time
    import csv
    logfile = open(args.save_dir + '/log.csv', 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'val_loss', 'val_acc'])
    logwriter.writeheader()

    t0 = time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    best_val_acc = 0.

    for epoch in range(args.epochs):
        model.train()  # set to training mode
        lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        ti = time()
        training_loss = 0.0
        for i, (x, y) in enumerate(train_loader):  # batch training
            y = torch.zeros(y.size(0), num_classes).scatter_(1, y.view(-1, 1), 1.)  # change to one-hot coding
            x, y = Variable(x.cuda()), Variable(y.cuda())  # convert input data to GPU Variable

            optimizer.zero_grad()  # set gradients of optimizer to zero
            y_pred, x_recon = model(x, y)  # forward
            loss = caps_loss(y, y_pred, x, x_recon, args.lam_recon)  # compute loss
            loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
            training_loss += loss.item() * x.size(0)#.data[0]  # record the batch loss
            optimizer.step()  # update the trainable parameters with computed gradients



        # compute validation loss and acc
        val_loss, val_acc = test(model, test_loader, args, num_classes)
        logwriter.writerow(dict(epoch=epoch, loss=training_loss / len(train_loader.dataset),
                                val_loss=val_loss, val_acc=val_acc))
        print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
              % (epoch, training_loss / len(train_loader.dataset),
                 val_loss, val_acc, time() - ti))
        if val_acc > best_val_acc:  # update best validation acc and save model
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)
            print("best val_acc increased to %.4f" % best_val_acc)


    list_metrics_clsf = []
    list_metrics_clsf.append([args.subj, val_acc])
    list_metrics_clsf = np.array(list_metrics_clsf)

    f = open("./outputs/accuracy-results-capsnet.csv", 'ab')
    np.savetxt(f, list_metrics_clsf, delimiter=",", fmt='%0.4f')
    f.close()

    logfile.close()
    torch.save(model.state_dict(), args.save_dir + '/trained_model.pkl')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    print("Total time = %ds" % (time() - t0))
    print('End Training' + '-' * 70)
    return model

