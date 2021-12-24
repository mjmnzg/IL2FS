#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import sys
import copy
import argparse
from modules import Data, sampling
from sklearn.model_selection import StratifiedShuffleSplit
from capsulenet import CapsuleNet
from utils.stratified_batch_sampler import StratifiedBatchSampler

try:
    import cPickle as pickle
except:
    import pickle

import modified_linear
from utils.compute_features import compute_features
from utils.compute_accuracy import compute_accuracy
from utils.incremental_train_and_eval_MR_LF import incremental_train_and_eval_MR_LF
import random

######### Modifiable Settings ##########
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--num_classes', default=33, type=int)
parser.add_argument('--num_output_capsules', default=16, type=int)
parser.add_argument('--nn', default="capsnet", type=str, help='lwnn for CL | capsnet or crnn for DREAMER and DEAP')
# number of classes in the first group
parser.add_argument('--nb_cl_fg', default=2, type=int, help='the number of classes in first group')
# number of class per group until to achieve total of the number of classes
parser.add_argument('--nb_cl', default=1, type=int, help='Classes per group')
parser.add_argument('--lam_recon', default=0.0005 * 784, type=float, help="The coefficient for the loss of decoder")
parser.add_argument('--nb_protos', default=30, type=int, help='Number of prototypes per class at the end')
parser.add_argument('--is_alg', default="herding", type=str, help='Instance selection algorithm')
parser.add_argument('--nb_runs', default=1, type=int, help='Number of runs (random ordering of classes at each run)')
parser.add_argument('--ckp_prefix', default=os.path.basename(sys.argv[0])[:-3], type=str, help='Checkpoint prefix')
parser.add_argument('--epochs', default=50, type=int, help='Epochs')
parser.add_argument('--T', default=2, type=float, help='Temporature for distialltion')
parser.add_argument('--beta', default=0.25, type=float, help='Beta for distialltion')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--fix_budget', default=False, type=bool, help='fix budget: True or False')
parser.add_argument('--subject', default=1, type=int, help='Number of subject')
parser.add_argument('--preprocess_format', default="rnn", type=str, help='rnn or cnn')
parser.add_argument('--dimention', default='discrete_emotions', type=str, help='arousal/valence/dominance/discrete_emotions')
parser.add_argument('--seed', default=223, type=int, help='random seed')
args = parser.parse_args()

# Set random SEED
np.random.seed(args.seed)
random.seed(args.seed)


train_batch_size       = 10            # Batch size for train
test_batch_size        = 50            # Batch size for test
eval_batch_size        = 50            # Batch size for eval
custom_weight_decay    = 5e-4           # Weight Decay
custom_momentum        = 0.9            # Momentum
args.ckp_prefix        = '{}_nb_cl_fg_{}_nb_cl_{}_nb_protos_{}'.format(args.ckp_prefix, args.nb_cl_fg, args.nb_cl, args.nb_protos)
np.random.seed(args.seed)        # Fix the random seed
print(args)
########################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Delimiter init into save file
f = open("./outputs/accuracy-results-" + args.dataset + "-" + args.is_alg + ".csv", 'ab')
np.savetxt(f, np.array([[0]]), delimiter=",", fmt='%0.4f')
f.close()

if args.dataset == 'DREAMER':

    # Set hyper-parameters for training
    base_lr = 0.001  # Initial learning rate | updated learning 0.001
    args.epochs = 50  # number of epochs
    lr_strat = [20, 30, 50]  # Epochs where learning rate gets decreased
    lr_factor = 0.1  # Learning rate decrease factor
    num_feature_layer = -2  # Layer to extract features
    data_file = str(args.subject - 1)

    # Information related with binary or multi-class problem
    dimension = args.dimention
    args.num_classes = 9
    args.num_output_capsules = 16
    dictionary_size = 108 # 108 for kfcv; 96 for train/testing
    args.use_walign = True

    with_or_without = 'yes'
    cnn_suffix = "_cnn_dataset.pkl"
    rnn_suffix = "_rnn_dataset.pkl"
    label_suffix = "_labels.pkl"
    type_data = "dreamer_mean"
    dataset_dir = "/home/magdiel/Descargas/Datasets/DREAMER/" + type_data + "/" + with_or_without + "_" + dimension + "/"

    # Load Samples
    # Format CNN
    if args.preprocess_format == 'cnn':
        with open(dataset_dir + data_file + cnn_suffix, "rb") as fp:
            dataset = pickle.load(fp)
        dataset = dataset.reshape(len(dataset), 128, 9, 9, 1)

    # Format RNN
    elif args.preprocess_format == 'rnn':
        with open(dataset_dir + data_file + rnn_suffix, "rb") as fp:
            dataset = pickle.load(fp)
        dataset = dataset.reshape(len(dataset), 128, 14, 1)

    # Load Labels
    with open(dataset_dir + data_file + "_" + dimension + label_suffix, "rb") as fp:
        labels = pickle.load(fp)
        labels = np.transpose(labels)


    # Stratified split
    s = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.seed)
    for train_index, test_index in s.split(dataset, labels):
        Sx, Tx = dataset[train_index], dataset[test_index]
        Sy, Ty = labels[train_index], labels[test_index]

    print("Sx-shape:", Sx.shape, " Sy-shape:", Sy.shape, " Tx-shape:", Tx.shape, " Ty-shape:", Ty.shape)
    print("Labels:", np.unique(Sy))



elif args.dataset == 'DEAP':

    # Set hyper-parameters for dataset
    base_lr = 0.001  # Initial learning rate | updated learning 0.001
    args.epochs = 20  # number of epochs
    lr_strat = [10, 15]  # Epochs where learning rate gets decreased
    lr_factor = 0.1  # Learning rate decrease factor
    num_feature_layer = -2  # Layer to extract features
    data_file = 's%02d'%args.subject

    dimension = args.dimention
    args.num_output_capsules = 16
    args.use_walign = True
    args.mnemonics_total_epochs = 10

    with_or_without = 'yes'
    cnn_suffix = ".mat_win_128_cnn_dataset.pkl"
    rnn_suffix = ".mat_win_128_rnn_dataset.pkl"
    label_suffix = ".mat_win_128_labels.pkl"
    type_data = "deap_eight"
    dataset_dir = "/home/magdiel/Descargas/Datasets/DEAP/" + type_data + "/" + with_or_without + "_" + dimension + "/"

    # Load training set for CNN
    # Format RNN
    if args.preprocess_format == 'cnn':
        with open(dataset_dir + data_file + cnn_suffix, "rb") as fp:
            dataset = pickle.load(fp)
        dataset = dataset.reshape(len(dataset), 128, 9, 9, 1)
    elif args.preprocess_format == 'rnn':
        with open(dataset_dir + data_file + rnn_suffix, "rb") as fp:
            dataset = pickle.load(fp)
        dataset = dataset.reshape(len(dataset), 128, 32, 1)

    # Load Labels
    with open(dataset_dir + data_file + label_suffix, "rb") as fp:
        labels = pickle.load(fp)
        labels = np.transpose(labels)


    args.num_classes = len(np.unique(labels))  # number of classes

    # obtain labels
    order_list = list(np.unique(labels))
    labels = np.array([order_list.index(i) for i in labels])

    # Stratified split
    s = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    for train_index, test_index in s.split(dataset, labels):
        Sx, Tx = dataset[train_index], dataset[test_index]
        Sy, Ty = labels[train_index], labels[test_index]

    lbls = np.unique(Sy)

    # obtain the maximum number of samples
    maximum = -1
    for l in lbls:
        selected_set = np.where(Sy == l)
        size = len(selected_set[0])
        if maximum < size:
            maximum = size

    print("dictionary_size:", maximum)
    dictionary_size = maximum
    args.num_classes = len(lbls)

    # [Sampling data]
    Sx, Sy = sampling(Sx, Sy, dictionary_size, args.num_classes, args)

    print("Sx-shape:", Sx.shape, " Sy-shape:", Sy.shape, " Tx-shape:", Tx.shape, " Ty-shape:", Ty.shape)
    print("Labels:", np.unique(Sy))

    # determine number of prototypes to store
    args.nb_protos = int((len(Sx) * args.nb_protos) / 100)
    print("Size memory:", args.nb_protos)

# Backup dataset because trainset and testset are updated for the training
X_train_total = np.array(Sx)
Y_train_total = np.array(Sy)
X_valid_total = np.array(Tx)
Y_valid_total = np.array(Ty)
input_dim = Sx.shape[1]

print("Training data size:", X_train_total.shape)
print("Test data size:", X_valid_total.shape)


# Launch the different runs
for iteration_total in range(args.nb_runs):

    # Generate list with order of classes
    print("Generating orders")
    order_name = "./checkpoint/seed_{}_{}_order_run_{}.pkl".format(args.seed, args.dataset, iteration_total)
    order = np.arange(args.num_classes)

    # DELETE FOR CLASSIFICATION of ALL CLASSES
    np.random.shuffle(order)

    # [SAVE_COMMENTED]
    #utils_pytorch.savepickle(order, order_name)

    order_list = list(order)
    print("ORDER LIST:", order_list)


    # Initialization of the variables for this run
    X_valid_cumuls    = []
    X_protoset_cumuls = []
    X_train_cumuls    = []
    Y_valid_cumuls    = []
    Y_protoset_cumuls = []
    Y_train_cumuls    = []

    alpha_dr_herding  = np.zeros((int(args.num_classes/args.nb_cl), dictionary_size, args.nb_cl), np.float32)

    # The following contains all the training samples of the different classes
    # because we want to compare our method with the theoretical case where all the training samples are stored
    # *** Store in prototypes the data considering chunks of classes

    # Build shape for prototype tensor
    shape_a = np.array([args.num_classes, dictionary_size])
    shape_b = np.array(X_train_total.shape)[1:]
    array_shape = np.concatenate((shape_a, shape_b), axis=0).astype(int)
    # build tensor for samples
    prototypes = np.zeros(array_shape)
    # build tensor for labels
    prototypes_y = np.zeros((args.num_classes, dictionary_size))
    for orde in range(args.num_classes):
        prototypes[orde, :, :] = X_train_total[np.where(Y_train_total == order[orde])]
        prototypes_y[orde, :] = Y_train_total[np.where(Y_train_total == order[orde])]

    # nb_cl_fg: the number of classes in first group
    # nb_cl: Classes per group
    start_iter = int(args.nb_cl_fg / args.nb_cl) - 1
    print("ITERATIONS - start_iter:", start_iter, " final:", int(args.num_classes / args.nb_cl))


    for iteration in range(start_iter, int(args.num_classes / args.nb_cl)):

        start_time = time.time()

        # FIRST ITERATION (BASE MODEL TRAINED WITH A GROUP OF CLASSES)
        if iteration == start_iter:
            ############################################################
            last_iter = 0
            ############################################################

            if args.dataset in ['DREAMER', 'DEAP']:
                tg_model = CapsuleNet(input_size=Sx.shape[1:], num_output_capsules=args.num_output_capsules, classes=args.nb_cl_fg, routings=3)
                tg_model.cuda()

                # discriminator network
                feature_discriminator = None

            # Get dimensionality for the consine layer
            in_features = tg_model.fc.in_features
            # Get dimensionality for the output layer (classes)
            out_features = tg_model.fc.out_features # number of classes
            print("in_features:", in_features, "out_features:", out_features)
            ref_model = None
            free_model = None
            ref_free_model = None

        # SECOND ITERATION (add CONSINE LAYER)
        elif iteration == start_iter+1:

            if args.dataset in ['DREAMER']:
                args.epochs = 50
                base_lr = 0.001
                lr_strat = [30, 40, 50]

            elif args.dataset in ['DEAP']:
                args.epochs = 20
                base_lr = 0.001
                lr_strat = [10, 15, 20]

            ############################################################
            last_iter = iteration # number of iteration
            ############################################################
            #increment classes
            # reference model (previous)
            ref_model = copy.deepcopy(tg_model)
            # set new dimensionality for feature and output layers
            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            print("in_features:", in_features, "out_features1:", out_features)

            # ADD CONSINE LAYER (size input, output, number classes in this stage)
            # CREATE A NEW LAYER with the finish number of classes
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features, args.nb_cl)

            # reasign outputs
            new_fc.fc1.weight.data = tg_model.fc.weight.data
            new_fc.sigma.data = tg_model.fc.sigma.data
            tg_model.fc = new_fc


        # ADD NEW LAYER in each iteration
        else:

            if args.dataset in ['DREAMER']:
                args.epochs = 50
                base_lr = 0.001
                lr_strat = [30, 40, 50]

            elif args.dataset in ['DEAP']:
                args.epochs = 20
                base_lr = 0.001
                lr_strat = [10, 15, 20]

            ############################################################
            last_iter = iteration # set iteration
            ############################################################

            # model reference (copy)
            ref_model = copy.deepcopy(tg_model)

            # get feature and output layers
            in_features = tg_model.fc.in_features
            out_features1 = tg_model.fc.fc1.out_features
            out_features2 = tg_model.fc.fc2.out_features

            print("in_features:", in_features, "out_features1:", out_features1, "out_features2:", out_features2)
            # ADD CONSINE LAYER (FOR NEW FEATURES)
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features1 + out_features2, args.nb_cl)
            new_fc.fc1.weight.data[:out_features1] = tg_model.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = tg_model.fc.fc2.weight.data
            new_fc.sigma.data = tg_model.fc.sigma.data
            tg_model.fc = new_fc


        # Prepare the training data for the current batch of classes
        actual_cl = order[range(last_iter*args.nb_cl, (iteration+1)*args.nb_cl)]
        print("LIST:", order)
        print("SELECTED:", actual_cl)
        # EXTRACT data of each group (creates array with Trues for selected samples)
        indices_train_10 = np.array([i in order[range(last_iter*args.nb_cl, (iteration+1)*args.nb_cl)] for i in Y_train_total])
        indices_test_10  = np.array([i in order[range(last_iter*args.nb_cl, (iteration+1)*args.nb_cl)] for i in Y_valid_total])


        print("GET DATA according to Order List")


        # get data samples
        X_train          = X_train_total[indices_train_10]
        X_valid          = X_valid_total[indices_test_10]
        X_valid_cumuls.append(X_valid)
        X_train_cumuls.append(X_train)
        X_valid_cumul    = np.concatenate(X_valid_cumuls)
        X_train_cumul    = np.concatenate(X_train_cumuls)

        # get data labels
        Y_train = Y_train_total[indices_train_10]
        Y_valid = Y_valid_total[indices_test_10]
        Y_valid_cumuls.append(Y_valid)
        Y_train_cumuls.append(Y_train)
        Y_valid_cumul = np.concatenate(Y_valid_cumuls)
        Y_train_cumul = np.concatenate(Y_train_cumuls)


        print("COMING IN TRAINING DATA:", X_train.shape, Y_train.shape)
        print("COMING IN TEST DATA:", X_valid.shape, Y_valid.shape)

        # Store original data (validation)
        if iteration == start_iter:
            # if it is the first iteration, then it is the original valid data
            X_valid_ori = X_valid
            Y_valid_ori = Y_valid
        else:

            # Concatenate acumulated prototypes
            X_protoset = np.concatenate(X_protoset_cumuls)
            Y_protoset = np.concatenate(Y_protoset_cumuls)

            print("OLD SAMPLES:", X_protoset.shape, Y_protoset.shape)

            # Concatenate [New samples + old samples]
            X_train = np.concatenate((X_train, X_protoset), axis=0)
            Y_train = np.concatenate((Y_train, Y_protoset))

            print("NEW SAMPLES + OLD SAMPLES:", X_train.shape, Y_train.shape)


        # TRAINING LOOP
        print('ITERATION ...', str(iteration+1))
        # store classes that will be evaluated
        #   from current batch
        map_Y_train = np.array([order_list.index(i) for i in Y_train])
        #   from all cumulative data
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])

        # UPDATE trainset (It is used to train the model with the number of classes available)
        trainset = Data(X_train, map_Y_train)

        list_cls = np.unique(map_Y_train)
        num_current_classes = len(list_cls)

        # OBTAIN num samples for each class
        cls_num_list = []
        for h in range(num_current_classes):
            a1 = np.where(map_Y_train == list_cls[h])
            cls_num_list.append(a1[0].shape[0])

        # Dataloader
        trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=StratifiedBatchSampler(map_Y_train, batch_size=train_batch_size, num_classes=num_current_classes), num_workers=2) #batch_size=train_batch_size, shuffle=True, num_workers=2)


        # UPDATE testset
        testset = Data(X_valid_cumul, map_Y_valid_cumul)
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)


        # Range of datasets coming in to training
        print('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
        print('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid_cumul), max(map_Y_valid_cumul)))

        ##############################################################
        ckp_name = './checkpoint/{}_run_{}_iteration_{}_model.pth'.format(args.ckp_prefix, iteration_total, iteration)
        print('ckp_name', ckp_name)

        # [Train model]
        if args.resume and os.path.exists(ckp_name):
            # Trained model
            print("###############################")
            print("Loading models from checkpoint")
            tg_model = torch.load(ckp_name)
            print("###############################")
        else:
            ###############################
            tg_params = tg_model.parameters()

            ###############################
            # assign target model to device
            tg_model = tg_model.to(device)
            # assign reference model to device
            if iteration > start_iter:
                ref_model = ref_model.to(device)

            # Set optimizer
            tg_optimizer = optim.Adam(tg_params, lr=base_lr, weight_decay=custom_weight_decay)

            # set scheduler for optimizer
            tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)


            # Train deep network model
            # NOTE: the type of training depends of selected options
            ###############################
            print("incremental_train_and_eval_MR_LF")
            # hyper-parameters
            #   epochs
            #   tg_model: target model
            #   ref_model: reference model
            #   tg_optimizer: optimizer used to train model
            #   tg_lr_scheduler: schedule for learning rate
            #   trainloader: training data
            #   testloader: test data
            #   iteration: iteration
            #   start_iter: initial iteration
            #   cur_lambda: weight factor
            #   args.dist, args.K, args.lw_mr: hyper-parameters for loss function
            tg_model = incremental_train_and_eval_MR_LF(args.epochs, tg_model, ref_model, tg_optimizer,
                                                        tg_lr_scheduler,
                                                        trainloader,
                                                        testloader,
                                                        iteration,
                                                        start_iter,
                                                        args=args)

            # create directory for checkpoint
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            # save checkpoint
            # :COMMENTED
            #torch.save(tg_model, ckp_name)

        ### Exemplars
        if args.fix_budget:
            nb_protos_cl = int(np.ceil(args.nb_protos / num_current_classes))
            print("[FIXED MEMORY]")
            print("SAMPLES PER CLASS:", nb_protos_cl)

        else:
            nb_protos_cl = args.nb_protos

        # container target feature model
        # [NOTE: I change -1 to -2]
        tg_feature_model = nn.Sequential(*list(tg_model.children())[:num_feature_layer])
        print("Sequential model:", tg_feature_model)

        # GET the number of features
        num_features = tg_model.fc.in_features

        print("--- %s seconds ---" % (time.time() - start_time))


        ################################################
        # Select the exemplar set (samples from the OLD DATASET)
        ################################################
        print('Updating exemplar set...')

        # +++++++++++ HERDING ++++++++++++++++++++++
        # Select exemplars in the protoset
        # It iter over number of classes whithin of each group

        for iter_dico in range(last_iter*args.nb_cl, (iteration+1)*args.nb_cl):
            # Possible exemplars in the feature space and projected on the L2 sphere
            # Get data for new class
            # In each iteration selects prototype candidates
            Tx = prototypes[iter_dico]
            Ty = np.zeros(Tx.shape[0])
            ###############################################################
            evalset = Data(Tx, Ty)
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
            num_samples = Tx.shape[0]

            ###############################################################

            mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
            # L2 norm = V / ||V||
            D = mapped_prototypes.T         # transpose
            D = D/(np.linalg.norm(D,axis=0) + 0.0001)   # normalize

            ###############################################################################
            # Herding procedure : ranking potential exemplars
            ###############################################################################
            # samples mean from class
            mu = np.mean(D,axis=1)              # samples mean
            # iterate over the stage in alpha_dr_herding [stage, # samples, class of the group]
            index1 = int(iter_dico/args.nb_cl)
            # iterate over the class of the group in alpha_dr_herding [stage, # samples, class of the group]
            index2 = iter_dico % args.nb_cl     # end index (residual)
            alpha_dr_herding[index1, :, index2] = alpha_dr_herding[index1, :, index2] * 0 # init alpha herding
            w_t = mu
            iter_herding = 0
            iter_herding_eff = 0

            while not(np.sum(alpha_dr_herding[index1, :, index2] != 0) == min(nb_protos_cl, 50)) and iter_herding_eff < 1000:
                tmp_t = np.dot(w_t, D)
                ind_max = np.argmax(tmp_t)
                iter_herding_eff += 1
                if alpha_dr_herding[index1, ind_max, index2] == 0:
                    alpha_dr_herding[index1, ind_max, index2] = 1 + iter_herding
                    iter_herding += 1

                w_t = w_t + mu - D[:, ind_max]
            ###############################################################################

        #################################################################################
        # Prepare the protoset
        X_protoset_cumuls = []
        Y_protoset_cumuls = []

        # Class means for iCaRL and NCM + Storing the selected exemplars in the protoset
        print('Computing mean-of_exemplars and theoretical mean...')
        # (num_features, number of classes (all), [NCM, NEM])
        class_means = np.zeros((num_features, 10, 2))

        for iteration2 in range(iteration+1): # iterar sobre la etapas
            for iter_dico in range(args.nb_cl): # iterar sobre las clases de cada etapa
                ############################################################################
                ###### IT IS USED TO UPDATE EXEMPLAR SET ########
                ########################################################################
                # class is obtained in step 'iteration2', according to order list
                current_cl = order[range(iteration2*args.nb_cl,(iteration2+1)*args.nb_cl)]
                ############################################################################

                # Collect data in the feature space
                Tx = prototypes[iteration2 * args.nb_cl + iter_dico]
                Ty = np.zeros(Tx.shape[0])
                evalset = Data(Tx, Ty)
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=2)

                # Get feature representation from given data
                num_samples = Tx.shape[0]
                mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)

                # L2 Norm
                D = mapped_prototypes.T
                D = D/(np.linalg.norm(D, axis=0)+0.00001)

                # Flipped version (features are reversed)
                # prototypes(args.num_classes, dictionary_size, X_train_total.shape[1])
                # >>> a = '1234'
                # >>> a[::-1]
                # '4321'
                evalset.data = prototypes[iteration2 * args.nb_cl + iter_dico][:, ::-1]
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
                mapped_prototypes2 = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                # L2 norm
                D2 = mapped_prototypes2.T
                D2 = D2/(np.linalg.norm(D2,axis=0) + 0.00001)

                # iCaRL
                #################################################################################
                # UPDATE EXEMPLAR SET
                #################################################################################
                #get alpha_dr_herding[incremental stage, number of samples (all), current class according to iterdico]
                alph = alpha_dr_herding[iteration2, :, iter_dico]
                # (elements of alpha greater than 0) * (elements of alpha lower than number of prototypes +1)
                # colocamos las banderas en 1 para los prototipos seleccionados
                alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.

                # gets the prototypes con las banderas en 1
                X_protoset_cumuls.append(prototypes[iteration2*args.nb_cl+iter_dico, np.where(alph==1)[0]])
                Y_protoset_cumuls.append(order[iteration2*args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))
                ##################################################################################

                # NMC
                alph = alph/np.sum(alph)
                # obtain mean for current class multiplying by the matrix
                class_means[:,current_cl[iter_dico],0] = (np.dot(D, alph) + np.dot(D2, alph))/2
                # L2 norm
                class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico], 0])


                # NEM (theoretical mean: all samples are used.)
                alph = np.ones(dictionary_size)/dictionary_size
                #print("Mean on all samples:", alph)
                class_means[:,current_cl[iter_dico], 1] = (np.dot(D, alph) + np.dot(D2, alph))/2
                class_means[:,current_cl[iter_dico], 1] /= np.linalg.norm(class_means[:, current_cl[iter_dico], 1])

                #################################
        current_means = class_means[:, order[range(0, (iteration + 1) * args.nb_cl)]]

        # set selected samples
        x_samples = np.concatenate(X_protoset_cumuls)
        y_samples = np.concatenate(Y_protoset_cumuls)


        print("Number of SELECTED EXAMPLERS:", x_samples.shape, y_samples.shape)

        evalset = Data(x_samples, y_samples)
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
        num_samples = x_samples.shape[0]

        ###############################################################

        mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
        x_samples = mapped_prototypes
        y_samples = np.array([order_list.index(i) for i in y_samples])

        # :COMMENTED
        #torch.save(class_means, './checkpoint/{}_run_{}_iteration_{}_class_means.pth'.format(args.ckp_prefix, iteration_total, iteration))
        #################################################################################

        ##############################################################
        # ACCURACY over initial classes
        ######################################

        # Calculate validation error of model on the first nb_cl classes:
        map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
        print('Computing accuracy on the original batch of classes...')
        Tx = X_valid_ori
        Ty = map_Y_valid_ori

        evalset = Data(Tx, Ty)
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
        ori_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader, output_file=args.dataset+"-"+args.is_alg,
                                   x_samples=x_samples, y_samples=y_samples, current_iteration=iteration, current_seed=args.seed,
                                   gen_tsne=False, subject=args.subject, args=args)


        ##############################################################
        # Calculate validation error of model on the cumul of classes:
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
        print('Computing cumulative accuracy...')
        print("TESTING DATA:",X_valid_cumul.shape)
        Tx = X_valid_cumul
        Ty = map_Y_valid_cumul
        evalset = Data(Tx, Ty)
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
        cumul_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader, output_file=args.dataset+"-"+args.is_alg,
                                     x_samples=x_samples, y_samples=y_samples, current_iteration=iteration, current_seed=args.seed,
                                     gen_tsne=False, subject=args.subject, args=args)



    # Final save of the data
    # :COMMENTED
    #torch.save(top1_acc_list_ori, \
    #    './checkpoint/{}_run_{}_top1_acc_list_ori.pth'.format(args.ckp_prefix, iteration_total))
    #torch.save(top1_acc_list_cumul, \
    #    './checkpoint/{}_run_{}_top1_acc_list_cumul.pth'.format(args.ckp_prefix, iteration_total))


