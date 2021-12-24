#!/usr/bin/env python
# coding=utf-8
from torch.autograd import Variable
from utils_pytorch import *
from capsulenet import caps_loss, test
from time import time

cur_features = []
ref_features = []
old_scores = []
new_scores = []
cur_features = []
ref_features = []

def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs



def incremental_train_and_eval_MR_LF(epochs, tg_model, ref_model, tg_optimizer,
                                     tg_lr_scheduler,
                                     trainloader,
                                     testloader,
                                     iteration,
                                     start_iteration,
                                     weight_per_class=None,
                                     device=None,
                                     args=None):

    # select device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # handle reference model
    if iteration > start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features
        handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
        handle_cur_features = tg_model.fc.register_forward_hook(get_cur_features)
        handle_old_scores_bs = tg_model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
        handle_new_scores_bs = tg_model.fc.fc2.register_forward_hook(get_new_scores_before_scale)

    # number of current classes
    num_current_classes = tg_model.fc.out_features


    for epoch in range(epochs):

        tg_model.train()

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        correct = 0
        total = 0
        tg_lr_scheduler.step()
        print('\nEpoch: %d, LR: ' % epoch, end='')
        print(tg_lr_scheduler.get_last_lr())

        ti = time()
        # First iteration (initial model)
        if iteration == start_iteration:  # only if it is the first group (base model)

            for batch_idx, (inputs, targets) in enumerate(trainloader):
            # TODO: Sampling [IMPORTANT]
            #for batch_idx, (inputs, targets, old_inputs, old_labels, new_inputs, new_labels) in enumerate(extendedDataLoader):

                # initialize gradients
                tg_optimizer.zero_grad()

                if args.nn in ['capsnet']:
                    # get one hot vector to extract outputs from capsules
                    y = torch.zeros(targets.size(0), args.num_output_capsules).scatter_(1, targets.view(-1, 1), 1.)
                    x, y = Variable(inputs.cuda()), Variable(y.cuda())  # convert input data to GPU Variable
                    # get outputs and reconstruction
                    outputs, x_recon = tg_model(x, y)  # forward
                    # get one hot vector to outputs from fully connected layers
                    y = torch.zeros(targets.size(0), num_current_classes).scatter_(1, targets.view(-1, 1), 1.)
                    y = Variable(y.cuda())  # convert input data to GPU Variable

                    loss = caps_loss(y, outputs, x, x_recon, args.lam_recon)

                    # to GPU device
                    inputs, targets = inputs.to(device), targets.to(device)

                else:
                    # to GPU device
                    inputs, targets = inputs.to(device), targets.to(device)
                    # get outputs
                    outputs = tg_model(inputs)
                    # main loss
                    loss = nn.CrossEntropyLoss()(outputs, targets)


                # Backpropagation
                loss.backward()
                tg_optimizer.step()

                # get value of loss
                if args.nn in ['capsnet']:
                    train_loss += loss.item() * x.size(0)
                else:
                    train_loss += loss.item()
                _, predicted = outputs.max(1)

                # LOSS VALUE
                total += targets.size(0)
                # PREDICTIONS
                correct += predicted.eq(targets).sum().item()

        else:

            for batch_idx, (inputs, targets) in enumerate(trainloader):

                ####################################################################
                # initialize gradients
                tg_optimizer.zero_grad()

                loss1 = None
                loss3 = None

                if args.nn in ['capsnet']:
                    # get one hot vector to extract outputs from capsules
                    y = torch.zeros(targets.size(0), args.num_output_capsules).scatter_(1, targets.view(-1, 1), 1.)
                    x, y = Variable(inputs.cuda()), Variable(y.cuda())  # convert input data to GPU Variable
                    # get outputs and reconstruction
                    outputs, x_recon = tg_model(x, y)  # forward
                    # get one hot vector to outputs from fully connected layers
                    cls = torch.zeros(targets.size(0), num_current_classes).scatter_(1, targets.view(-1, 1), 1.)
                    cls = Variable(cls.cuda())  # convert input data to GPU Variable

                    # ---------------
                    # [BASELINE LOSS]
                    # ---------------
                    loss = caps_loss(cls, outputs, x, x_recon, args.lam_recon)

                    # to GPU device
                    inputs, targets = inputs.to(device), targets.to(device)

                    # obtain reference outputs
                    ref_outputs, _ = ref_model(x, y)

                else:
                    inputs, targets = inputs.to(device), targets.to(device)
                    # get outputs
                    outputs = tg_model(inputs)
                    # cross entropy loss
                    loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

                    # obtain reference outputs
                    ref_outputs = ref_model(inputs)


                # BACK-PROPAGATION
                loss.backward()
                tg_optimizer.step()


                train_loss += loss.item()
                if iteration > start_iteration:
                    train_loss1 += loss.item()
                _, predicted = outputs.max(1)
                # LOSS VALUE
                total += targets.size(0)
                # PREDICTIONS
                correct += predicted.eq(targets).sum().item()

        if iteration == start_iteration:
            print('Train set: {}, Train Loss: {:.4f} Acc: {:.4f}'.format(\
                len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))
        else:
            print('Train set: {}, Train Loss: {:.4f} Acc: {:.4f}'.format( \
                len(trainloader), train_loss1 / (batch_idx + 1), 100. * correct / total))


        # Evaluate on test data
        tg_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        if args.nn in ['capsnet']:
            # compute validation loss and acc
            val_loss, val_acc = test(tg_model, testloader, args, num_current_classes)
            print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
                  % (epoch, train_loss / len(trainloader.dataset),
                     val_loss, val_acc, time() - ti))

        else:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):

                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = tg_model(inputs)
                    loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            print('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format(\
                len(testloader), test_loss/(batch_idx+1), 100.*correct/total))


    if iteration > start_iteration:
        print("Removing register_forward_hook")
        handle_ref_features.remove()
        handle_cur_features.remove()
        handle_old_scores_bs.remove()
        handle_new_scores_bs.remove()

    # return trained model
    return tg_model