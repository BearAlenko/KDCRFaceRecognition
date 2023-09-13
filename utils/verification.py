"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from signal import Sigmasks
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import sklearn
import enum
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import copy
import torch
import torch.nn as nn
from datetime import date
from numpy.linalg import norm
from scipy.spatial.distance import cdist



# Support: ['calculate_roc', 'calculate_accuracy', 'calculate_val', 'calculate_val_far', 'evaluate']


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds = 10, pca = 0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits = nrof_folds, shuffle = False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
        # dist = pdist(np.vstack([embeddings1, embeddings2]), 'cosine')

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca > 0:
            print("doing pca on", fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis = 0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components = pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)
            #dist = cdist(embed1, embed2, 'cosine')

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
#         print('best_threshold_index', best_threshold_index, acc_train[best_threshold_index])
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def evaluate(embeddings1, embeddings2, actual_issame, nrof_folds = 10, pca = 0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), nrof_folds = nrof_folds, pca = pca)
#     thresholds = np.arange(0, 4, 0.001)
#     val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
#                                       np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
#     return tpr, fpr, accuracy, best_thresholds, val, val_std, far
    return tpr, fpr, accuracy, best_thresholds

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

def test_model(testloader, model, flip, model_no, model_name):
    model.eval()
    em_dim = 512
    embeddings1 = np.empty((0,em_dim))
    embeddings2 = np.empty((0,em_dim))
    labels_true = np.empty((0))
    progress = 0
    progress_checker = int(len(testloader)/10)
    with torch.no_grad():
        for i, inputs in enumerate(testloader):  
            progress += 1    
            if progress%progress_checker == 0:
                print(f'{int(progress/progress_checker)}'+'|..|'*int(progress/progress_checker))
            image0 = inputs['image0'].to(device)
            image1 = inputs['image1'].to(device)
            id_out0 = model(image0)
            id_out1 = model(image1)
            if type(id_out0) is tuple:
                id_out0 = id_out0[0]
                id_out1 = id_out1[0]
            if flip:
                flipped_image0 = torch.flip(image0, dims=(2,))
                flipped_image1 = torch.flip(image1, dims=(2,))
                id_flipout0 = model(flipped_image0)
                id_flipout1 = model(flipped_image1)
                if type(id_out0) is tuple:
                    id_flipout0 = id_flipout0[0]
                    id_flipout1 = id_flipout1[0]
                id_out0 = torch.cat((id_out0, id_flipout0), 1)
                id_out1 = torch.cat((id_out1, id_flipout1), 1)
            
            id_out0 = l2_norm(id_out0, axis=1).cpu().detach().numpy()
            id_out1 = l2_norm(id_out1, axis=1).cpu().detach().numpy()
            embeddings1 = np.concatenate((embeddings1, id_out0), axis=0)
            embeddings2 = np.concatenate((embeddings2, id_out1), axis=0)
            #cal_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
            #result = cal_similarity(id_out0, id_out1)
            #sims += result.detach().cpu()
            labels_true = np.concatenate((labels_true, inputs['label']), axis=0)
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings1=embeddings1, embeddings2= embeddings2,actual_issame=labels_true, nrof_folds = 10, pca = 0)
    #print(f'True positive rate is {str(tpr)}')
    #print(f'False positive rate is {str(fpr)}')
    print(f'Accuracy is {str(accuracy)}')
    print(f'True best threshold is {str(best_thresholds)}')
    th = np.mean(best_thresholds)
    print(f'Use threshold {str(th)} for large_test')
    return th

def test_model_large(testloader, model, flip, th):
    model.eval()
    sims = []
    labels_true = []
    progress = 0
    progress_checker = int(len(testloader)/10)
    with torch.no_grad():
        for i, inputs in enumerate(testloader):  
            progress += 1    
            if progress%progress_checker == 0:
                print(f'{int(progress/progress_checker)}'+'|..|'*int(progress/progress_checker))
            image0 = inputs['image0'].to(device)
            image1 = inputs['image1'].to(device)
            id_out0 = model(image0)
            id_out1 = model(image1)
            if type(id_out0) is tuple:
                id_out0 = id_out0[0]
                id_out1 = id_out1[0]
            if flip:
                flipped_image0 = torch.flip(image0, dims=(2,))
                flipped_image1 = torch.flip(image1, dims=(2,))
                id_flipout0 = model(flipped_image0)
                id_flipout1 = model(flipped_image1)
                if type(id_out0) is tuple:
                    id_flipout0 = id_flipout0[0]
                    id_flipout1 = id_flipout1[0]
                id_out0 = torch.cat((id_out0, id_flipout0), 1)
                id_out1 = torch.cat((id_out1, id_flipout1), 1)
            
            id_out0 = l2_norm(id_out0, axis=1).cpu().detach().numpy()
            id_out1 = l2_norm(id_out1, axis=1).cpu().detach().numpy()
            diff = np.subtract(id_out0, id_out1)
            dist = np.sum(np.square(diff), 1)
            sims += dist.tolist()
            labels_true += inputs['label']
    test_correct = 0
    print(sims)
    for i, sim in enumerate(sims):
        if (sim <= th)==labels_true[i]:
            test_correct += 1
    acc = test_correct/len(sims)
    print(f'Accuracy of large age gap is {str(acc)}')
