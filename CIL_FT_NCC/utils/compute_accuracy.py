#!/usr/bin/env python
# coding=utf-8
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.distance import cdist
from utils_pytorch import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestCentroid


def compute_accuracy(tg_model,
                     tg_feature_model,
                     class_means,
                     evalloader,
                     scale=None,
                     print_info=True,
                     device=None,
                     output_file="",
                     x_samples=None,
                     y_samples=None,
                     current_iteration=0,
                     current_seed=0,
                     gen_tsne=False,
                     subject=None,
                     args=None):

    # set device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()
    tg_feature_model.eval()

    correct = 0
    correct_icarl = 0
    correct_ncm = 0
    total = 0

    y_pred = None
    y_test = None
    x_outs = None
    x_feats = None

    nc_clf = NearestCentroid(shrink_threshold=None)
    nc_clf.fit(x_samples, y_samples)



    flag = False
    nc_preds = None
    y_true = None

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):

            # Compute score for CNN
            if args.nn in ['capsnet']:
                # this is used as complement but it is not useful
                y = torch.zeros(targets.size(0), args.num_output_capsules).scatter_(1, targets.view(-1, 1), 1.)
                y = Variable(y.cuda())

                inputs, targets = inputs.to(device), targets.to(device)
                total += targets.size(0)
                outputs, _ = tg_model(inputs, y)
            else:
                inputs, targets = inputs.to(device), targets.to(device)
                total += targets.size(0)
                outputs = tg_model(inputs)

            outputs = F.softmax(outputs, dim=1)

            # for plot tsne
            tsne_outs = outputs.cpu().numpy()

            if scale is not None:
                assert(scale.shape[0] == 1)
                assert(outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            # get features
            outputs_feature = tg_feature_model(inputs)
            outputs_feature = np.squeeze(outputs_feature)

            #######
            if batch_idx == 0:
                y_pred = predicted.cpu().numpy()
                y_test = targets.cpu().numpy()
                x_outs = tsne_outs
                x_feats = outputs_feature.cpu().numpy()
            else:
                y_pred = np.concatenate((y_pred, predicted.cpu().numpy()), axis=0)
                y_test = np.concatenate((y_test, targets.cpu().numpy()), axis=0)
                x_outs = np.concatenate((x_outs, tsne_outs), axis=0)
                x_feats = np.concatenate((x_feats, outputs_feature.cpu().numpy()), axis=0)
            #####

            # Compute score for iCaRL
            feats = outputs_feature.cpu().numpy()

            sqd_icarl = cdist(class_means[:,:,0].T, feats, 'sqeuclidean')
            score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
            _, predicted_icarl = score_icarl.max(1)
            correct_icarl += predicted_icarl.eq(targets).sum().item()

            # Compute score for NCM
            sqd_ncm = cdist(class_means[:,:,1].T, feats, 'sqeuclidean')
            score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
            _, predicted_ncm = score_ncm.max(1)
            correct_ncm += predicted_ncm.eq(targets).sum().item()


            ###############################################
            if not flag:
                # Computed score for NCM (sklearn)
                nc_preds = nc_clf.predict(feats)
                y_true = targets.cpu().numpy()
                flag = True
            else:
                nc_p = nc_clf.predict(feats)
                nc_preds = np.concatenate((nc_preds, nc_p),axis=0)
                y_true = np.concatenate((y_true, targets.cpu().numpy()), axis=0)

            #################################################

    cnn_acc = 100. * correct / total
    icarl_acc = 100.*correct_icarl/total

    ncm_acc = 100.*correct_ncm/total
    nc_acc = accuracy_score(y_true, nc_preds) * 100


    if print_info:
        print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(cnn_acc))
        print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(icarl_acc))
        print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(ncm_acc))
        print("  top 1 accuracy NCC             :\t\t{:.2f} %".format(nc_acc))


    # obtain unique labels
    labels = np.unique(y_test)
    labels_name = [str(i) for i in labels]

    # get confusion matrix
    matrix = confusion_matrix(y_test, y_pred)
    print("confusion_matrix:\n", matrix)

    list_metrics_clsf = []
    list_metrics_clsf.append([cnn_acc, icarl_acc, ncm_acc, nc_acc])
    list_metrics_clsf = np.array(list_metrics_clsf)

    f = open("./outputs/accuracy-results-" + output_file + ".csv", 'ab')
    np.savetxt(f, list_metrics_clsf, delimiter=",", fmt='%0.4f')
    f.close()


    cnn_acc = 100.*correct/total
    icarl_acc = 100.*correct_icarl/total
    ncm_acc = 100.*correct_ncm/total

    return [cnn_acc, icarl_acc, ncm_acc]
