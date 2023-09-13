import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import os
import math

cudnn.benchmark = True
torch.manual_seed(17)
import random
random.seed(17)
import numpy as np
np.random.seed(17)
from pytorch_metric_learning import distances, losses, miners, reducers, testers

from model.AICNN import *
from model.AICNN_oe import *
from model.AICNN_Uni import *
from model.AICNN_t import *
from model.AICNN_oe_t import *
from model.AICNN_Uni_t import *
from model.AdaTransLearning import *
from model.losses import *
from train.data_loading import *
from utils.helper_functions import *
from utils.verification import *
from train.training import *
import argparse



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_dataset(train_data_root, test_data_root, train_dir, val_dir, test_dir, test_large_dir, batch_size, img_size, num_workers=0):
    train_loader, val_loader, test_loader, test_large_loader, train_size, val_size, test_size, test_large_size = get_train_valid_loader(train_data_root, test_data_root,
                                                                                                   train_dir, val_dir,
                           test_dir,
                           test_large_dir,
                           batch_size,
                           True,
                           10,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=True, img_size=img_size)
    dataloaders = {'train':train_loader,
               'val':val_loader,
               'test':test_loader,
               'testlarge':test_large_loader
               }
    dataset_sizes = {'train': train_size,
          'val': val_size,
                 'test': test_size,
                 'testlarge': test_large_size
                 }
    print(f'The image number of training data: {dataset_sizes["train"]}')
    print(f'The image number of val data: {dataset_sizes["val"]}')
    print(f'The image number of test data: {dataset_sizes["test"]}')
    print(f'The image number of test_large data: {dataset_sizes["testlarge"]}')
    print('.......................')
    return dataloaders, dataset_sizes

def train(model_type, dataloaders, dataset_sizes, regularizer_name, backbone_pre, model_name, model_no, learning_rate, margin, scale, dataroot, Temperature=10, T2=2, lock_weights=False, loss_name='ElasticFace_backbone', loading_test=False, args=None): 
    dir_train = dataroot + '/train'
    train_id = os.listdir(dir_train)
    num_classes = len(train_id)
    model = model_type(num_classes=num_classes, margin=margin, scale=scale)
    model = model.cuda()
    num_epochs = args.num_epochs
    #l = 0.1
    #if regularizer_name == 'regular_face':
    #    regularizer = regularizers.RegularFaceRegularizer()
    #    idLoss = losses.ArcFaceLoss(num_classes = num_classes, embedding_size = 512, margin=margin, scale=scale, weight_regularizer=regularizer)
    #else:
    #    idLoss = losses.ArcFaceLoss(num_classes = num_classes, embedding_size = 512, margin=margin, scale=scale)
    #age_loss = nn.CrossEntropyLoss()
    
    
    #if model_name == 'AICNN_Uni' or 'AICNN_Uni_t':
    #    def criterion(id_features, age_out, labels, ages):
    #        loss = idLoss(id_features, labels)
    #        return loss
    #elif model_name == 'AICNN_oe' or 'AICNN_oe_t':
    #    l = 0.01
    #    def criterion(id_features, age_out, labels, ages):
    #        loss = idLoss(id_features, labels)+ l * age_loss(age_out, ages)
    #        return loss
    #else:
    #    def criterion(id_features, age_out, labels, ages):
    #        loss = idLoss(id_features, labels)+ l * age_loss(age_out, ages)
    #        return loss
    #optimizer_ft = optim.AdamW([{'params': model.parameters()},
    #                {'params': idLoss.parameters()}], lr=learning_rate)
    if loss_name == 'ElasticFace':
        classifier = nn.CrossEntropyLoss()
        def criterion(out, labels, ages):
            id_features = out[0]
            id_class = out[1]
            loss = classifier(id_class, labels)
            return loss
        optimizer_ft = optim.AdamW(model.parameters(), lr=learning_rate)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor = 0.5, patience = 5, verbose = 1) #from 1e-4

    elif loss_name == 'ElasticFace_backbone':
        model.load_init_backbone()
        classifier = nn.CrossEntropyLoss()
        def criterion(out, labels, ages):
            id_features = out[0]
            id_class = out[1]
            loss = classifier(id_class, labels)
            return loss
        optimizer_ft = optim.AdamW(model.head.parameters(), lr=learning_rate)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor = 0.5, patience = 5, verbose = 1) #from 1e-4
    elif loss_name == 'SoftFace_backbone':
        model.load_init_backbone()
        classifier = nn.CrossEntropyLoss()
        #age_classifier = nn.MSELoss()
        num_epochs=50
        def criterion(out, labels, ages):
            id_features = out[0]
            id_class = out[1]
            #age_out = out[2]
            #age = ages.float().unsqueeze(1)
            #loss = classifier(id_class+1e-8, labels) + 0.001*age_classifier(age_out, age)
            loss = classifier(id_class+1e-8, labels)
            return loss
        #optimizer_ft = optim.AdamW([{'params': model.backbone.age_trans.parameters()},
        #                            {'params': model.backbone.g.parameters()},
        #                            {'params': model.backbone.age_classify.parameters()},
        #                            {'params': model.head.parameters()}], lr=learning_rate)
        optimizer_ft = optim.AdamW(model.head.parameters(), lr=learning_rate)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor = 0.5, patience = 5, verbose = 1) #from 1e-4
    elif loss_name == 'AdaTransLearn_Elastic':
        model.load_init_backbone()
        kd = nn.CrossEntropyLoss()
        ce = nn.CrossEntropyLoss()
        def criterion(out, labels, ages):
            t_features = out[1]
            s_features = out[0]
            t_logits = out[3]
            s_logits = out[2]
            kd_loss = kd((s_logits/Temperature)+1e-8, nn.functional.softmax(t_logits/Temperature, dim=1))
            ce_loss = ce((s_logits/T2)+1e-8, labels)
            loss = kd_loss + ce_loss
            return loss, kd_loss, ce_loss
        optimizer_ft = optim.AdamW(model.student.parameters(), lr=learning_rate)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor = 0.5, patience = 5, verbose = 1, threshold = 0.02) #from 1e-4
    elif loss_name == 'AdaTransLearn_ElasticL2':
        model.load_init_backbone()
        kd = nn.CrossEntropyLoss()
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        def criterion(out, labels, ages):
            t_features = out[1]
            s_features = out[0]
            t_logits = out[3]
            s_logits = out[2]
            kd_loss = kd((s_logits/Temperature)+1e-8, nn.functional.softmax(t_logits/Temperature, dim=1))
            ce_loss = ce((s_logits/T2)+1e-8, labels)
            l2_loss = mse(s_features, t_features)
            loss = l2_loss + kd_loss + ce_loss
            return loss, kd_loss, ce_loss, l2_loss
        optimizer_ft = optim.AdamW(model.student.parameters(), lr=learning_rate)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor = 0.5, patience = 5, verbose = 1, threshold = 0.02) #from 1e-4
    elif loss_name == 'AdaTransLearn_ElasticTriL2':
        model.load_init_backbone()
        kd = nn.CrossEntropyLoss()
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        reducer = reducers.ThresholdReducer(low=0)
        distance = distances.LpDistance(power=2) #used to be cosine similarity
        tri_loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
        mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="all") #used to be 0.2 and semihard
        def criterion(out, labels, ages):
            t_features = out[1]
            s_features = out[0]
            t_logits = out[3]
            s_logits = out[2]
            indices_tuple = mining_func(s_features, labels)
            tri_loss = tri_loss_func(s_features, labels, indices_tuple)
            kd_loss = kd((s_logits/Temperature)+1e-8, nn.functional.softmax(t_logits/Temperature, dim=1))
            ce_loss = ce((s_logits/T2)+1e-8, labels)
            l2_loss = mse(s_features, t_features)
            loss = l2_loss + kd_loss + ce_loss + tri_loss
            return loss, kd_loss, ce_loss
        optimizer_ft = optim.AdamW(model.student.parameters(), lr=learning_rate)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor = 0.5, patience = 5, verbose = 1, threshold = 0.02) #from 1e-4
    elif loss_name == 'AdaTransLearn_ElasticTriL2Age':
        model.load_init_backbone()
        kd = nn.CrossEntropyLoss()
        ce = nn.CrossEntropyLoss()
        age = nn.MSELoss()
        mse = nn.MSELoss()
        reducer = reducers.ThresholdReducer(low=0)
        distance = distances.LpDistance(power=2) #used to be cosine similarity
        tri_loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
        mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="all") #used to be 0.2 and semihard
        def criterion(out, labels, ages):
            t_features = out[1]
            s_features = out[0]
            t_logits = out[3]
            s_logits = out[2]
            age_group = out[4]
            ages = ages.float().unsqueeze(1)
            indices_tuple = mining_func(s_features, labels)
            tri_loss = tri_loss_func(s_features, labels, indices_tuple)
            kd_loss = kd((s_logits/Temperature)+1e-8, nn.functional.softmax(t_logits/Temperature, dim=1))
            ce_loss = ce((s_logits/T2)+1e-8, labels)
            l2_loss = mse(s_features, t_features)
            age_loss = age(age_group, ages)
            loss = l2_loss + kd_loss + ce_loss + tri_loss + 0.001*age_loss
            return loss, kd_loss, ce_loss
        optimizer_ft = optim.AdamW(model.student.parameters(), lr=learning_rate)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor = 0.5, patience = 5, verbose = 1, threshold = 0.02)
    elif loss_name == 'AdaTransLearn_ArcL2':
        model.load_init_backbone()
        kd = nn.CrossEntropyLoss()
        model.student.head = ArcFace(in_features=512, out_features=num_classes, s=scale, m=margin)
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        def criterion(out, labels, ages):
            t_features = out[1]
            s_features = out[0]
            t_logits = out[3]
            s_logits = out[2]
            kd_loss = kd((s_logits/Temperature)+1e-8, nn.functional.softmax(t_logits/Temperature, dim=1))
            ce_loss = ce((s_logits/T2)+1e-8, labels)
            l2_loss = mse(s_features, t_features)
            loss = l2_loss + kd_loss + ce_loss
            return loss, kd_loss, ce_loss, l2_loss
        optimizer_ft = optim.AdamW(model.student.parameters(), lr=learning_rate)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor = 0.5, patience = 5, verbose = 1, threshold = 0.02) #from 1e-4
    elif loss_name == 'AdaTransLearn_ElasticTri':
        model.load_init_backbone()
        kd = nn.CrossEntropyLoss()
        ce = nn.CrossEntropyLoss()
        reducer = reducers.ThresholdReducer(low=0)
        distance = distances.LpDistance(power=2) #used to be cosine similarity
        tri_loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
        mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="all")
        def criterion(out, labels, ages):
            t_features = out[1]
            s_features = out[0]
            t_logits = out[3]
            s_logits = out[2]
            indices_tuple = mining_func(s_features, labels)
            tri_loss = tri_loss_func(s_features, labels, indices_tuple)
            kd_loss = kd((s_logits/Temperature)+1e-8, nn.functional.softmax(t_logits/Temperature, dim=1))
            ce_loss = ce((s_logits/T2)+1e-8, labels)
            loss = kd_loss + ce_loss + tri_loss
            return loss, kd_loss, ce_loss
        optimizer_ft = optim.AdamW(model.student.parameters(), lr=learning_rate)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor = 0.5, patience = 5, verbose = 1, threshold = 0.02) #from 1e-4
    elif loss_name == 'ElasticFace_trans':
        model.load_init_backbone()
        classifier = nn.CrossEntropyLoss()
        num_epochs=50
        def criterion(out, labels, ages):
            id_features = out[0]
            id_class = out[1]
            loss = classifier(id_class+1e-8, labels)
            return loss
        optimizer_ft = optim.AdamW(model.parameters(), lr=learning_rate)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor = 0.5, patience = 5, verbose = 1) #from 1e-4
    elif loss_name == 'SoftFace_trans':
        model.load_init_backbone()
        classifier = nn.CrossEntropyLoss()
        #age_classifier = nn.CrossEntropyLoss()
        #age_classifier = nn.MSELoss()
        def criterion(out, labels, ages):
            id_features = out[0]
            id_class = out[1]
            #age_out = out[2]
            #age = ages.float().unsqueeze(1)
            #loss = classifier(id_class+1e-8, labels) + 0.001*age_classifier(age_out, age)
            loss = classifier(id_class+1e-8, labels)
            return loss
        optimizer_ft = optim.AdamW(model.parameters(), lr=learning_rate)
        num_epochs = 50
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor = 0.5, patience = 5, verbose = 1) #from 1e-4
    elif loss_name == 'AdaTransLearn_ElasticL2_noKD':
        model.load_init_backbone()
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        def criterion(out, labels, ages):
            t_features = out[1]
            s_features = out[0]
            t_logits = out[3]
            s_logits = out[2]
            ce_loss = ce((s_logits/T2)+1e-8, labels)
            l2_loss = mse(s_features, t_features)
            loss = l2_loss + ce_loss
            return loss, ce_loss, l2_loss
        optimizer_ft = optim.AdamW(model.student.parameters(), lr=learning_rate)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor = 0.5, patience = 5, verbose = 1, threshold = 0.02) #from 1e-4
    elif loss_name == 'AdaTransLearn_ArcL2_noKD':
        model.load_init_backbone()
        model.student.head = ArcFace(in_features=512, out_features=num_classes, s=scale, m=margin)
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        idLoss = losses.ArcFaceLoss(num_classes = num_classes, embedding_size = 512, margin=math.degrees(margin), scale=scale)
        def criterion(out, labels, ages):
            t_features = out[1]
            s_features = out[0]
            t_logits = out[3]
            s_logits = out[2]
            ce_loss = ce((s_logits/T2)+1e-8, labels)
            l2_loss = mse(s_features, t_features)
            loss = l2_loss + ce_loss
            return loss, ce_loss, l2_loss
        optimizer_ft = optim.AdamW(model.student.parameters(), lr=learning_rate)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor = 0.5, patience = 5, verbose = 1, threshold = 0.02) #from 1e-4
    else:
        print('Incorrect loss name')
        return
    
    print(f'Has student in model {str(hasattr(model, "student"))}')
# Decay LR by a factor of 0.7 every 5 epochs
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs, model_name=model_name, runs=model_no, lock_weights=lock_weights, test=loading_test)
    save_model(model=model, model_name = model_name, model_no = model_no)
    return model

def find_threshold(dataloader, model, flip):
    model.eval()
    best_acc = 0
    sims = []
    labels_true = []
    progress = 0
    progress_checker = int(len(dataloader)/10)
    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            progress += 1
            if progress%progress_checker == 0:
                print(f'{int(progress/progress_checker)}'+'|..|'*int(progress/progress_checker))
            image0 = inputs['image0'].to(device)
            image1 = inputs['image1'].to(device)

            id_out0 = model(image0)
            id_out1 = model(image1)
            if flip:
                flipped_image0 = torch.flip(image0, dims=(2,))
                flipped_image1 = torch.flip(image1, dims=(2,))
                id_flipout0 = model(flipped_image0)
                id_flipout1 = model(flipped_image1)
                id_out0 = torch.cat((id_out0, id_flipout0), 1)
                id_out1 = torch.cat((id_out1, id_flipout1), 1)
            
            cal_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
            result = cal_similarity(id_out0, id_out1)
            sims += result.detach().cpu()
            labels_true += inputs['label']
        print(f'the length of similarity pairs is {len(sims)}')
    for th in sims:
        threshold = th
        test_correct = 0
        for i in range(len(sims)):
            sim = sims[i]
            if (sim >= threshold)==labels_true[i]:
                test_correct += 1
        acc = test_correct/len(sims)
        if acc > best_acc:
            best_acc = acc
            best_th = threshold
            print(f'current best acc = {best_acc}, th = {best_th}')
    print(f'{len(sims)} Val images: acc = {100 * best_acc} %, th = {best_th}')
    return best_th

def main():   
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--model", help="model name", default='AICNN')
    argParser.add_argument("--data_root", help="dataset dir", default = '/project/6003167/zzh2015/dataset/B3FD_filtered')
    argParser.add_argument("--model_no", help="model number", default = '1')
    argParser.add_argument("--train_mode", help="train or test", default = False, type = bool)
    argParser.add_argument("--batch_size", help="face similarity threshold", type = int, default=128)
    argParser.add_argument("--scale", help="face similarity threshold", type = float, default=64)
    argParser.add_argument("--margin", help="face similarity threshold", type = float, default=0.1)
    argParser.add_argument("--learning_rate", help="face similarity threshold", type = float, default=0.0002)
    argParser.add_argument("--img_size", help="face similarity threshold", type = int, default=112)
    argParser.add_argument("--flip_test", type = bool, default = False)
    argParser.add_argument("--not_find_th", type = bool, default = False)
    argParser.add_argument("--sim_th", type=float, default = -1.0)
    argParser.add_argument('--regularizer', default='regular_face', choices=['regular_face', 'none'])
    argParser.add_argument('--backbone_pre', default='none', choices=['vgg2', 'none'])
    argParser.add_argument("--lock_weights", type = bool, default = False)
    argParser.add_argument("--loss_name", default = 'ElasticFace_backbone', choices=['ElasticFace', 'ElasticFace_backbone', 'SoftFace_backbone', 'AdaTransLearn_Elastic', 'ElasticFace_trans', 'SoftFace_trans', 'AdaTransLearn_ElasticTri', 'AdaTransLearn_ElasticL2', 'AdaTransLearn_ElasticL2_noKD', 'AdaTransLearn_ArcL2', 'AdaTransLearn_ArcL2_noKD', 'AdaTransLearn_ElasticTriL2', 'AdaTransLearn_ElasticTriL2Age'])
    argParser.add_argument("--temperature", default = 10.0, type = float)
    argParser.add_argument("--temperature2", default = 1.0, type = float)
    argParser.add_argument("--num_workers", default = 0, type = int)
    argParser.add_argument("--loading_test", type = bool, default = False)
    argParser.add_argument("--num_epochs", default = 100, type = int)
    #argParser.add_argument("--weight_decay", type=float, default=5e-4)


    args = argParser.parse_args()
    
    model_name = args.model
    data_root = args.data_root
    train_mode = args.train_mode
    model_no = args.model_no
    batch_size= args.batch_size
    scale=args.scale
    margin=args.margin
    learning_rate=args.learning_rate
    img_size = args.img_size
    flip_test = args.flip_test
    find_th = not args.not_find_th
    sim_th = args.sim_th
    regularizer_name = args.regularizer
    backbone_pre = args.backbone_pre
    lock_weights = args.lock_weights
    loss_name = args.loss_name
    temperature = args.temperature
    temperature2 = args.temperature2
    num_workers = args.num_workers
    loading_test = args.loading_test

    model_type = AICNN
    if model_name == 'AICNN':
        model_type = AICNN
    if model_name == 'AICNN_oe':
        model_type = AICNN_oe
    if model_name == 'AICNN_Uni':
        model_type = AICNN_Uni
    if model_name == 'AICNN_t':
        model_type = AICNN_t
    if model_name == 'AICNN_oe_t':
        model_type = AICNN_oe_t
    if model_name == 'AICNN_Uni_t':
        model_type = AICNN_Uni_t
    if model_name == 'ElasticFace_backbone':
        model_type = ElasticFace_backbone
    if model_name == 'ElasticFace_backbone_ResNet34':
        model_type = ElasticFace_backbone_ResNet34
    if model_name == 'ElasticFace_backbone_ResNet50':
        model_type = ElasticFace_backbone_ResNet50
    if model_name == 'ElasticFace_backbone_ResNet100':
        model_type = ElasticFace_backbone_ResNet100
    if model_name == 'AdaTransLearn_Elastic':
        model_type = AdaTransLearn_Elastic
    if model_name == 'AdaTransLearn_Elastic_raw':
        model_type = AdaTransLearn_Elastic_raw
    if model_name == 'AdaTransLearn_Elastic_raw50':
        model_type = AdaTransLearn_Elastic_raw50
    if model_name == 'AdaTransLearn_Elastic_raw34':
        model_type = AdaTransLearn_Elastic_raw34
    if model_name == 'AdaTransLearn_Elastic_raw50FT':
        model_type = AdaTransLearn_Elastic_raw50FT
    if model_name == 'AdaTransLearn_Soft_raw50':
        model_type = AdaTransLearn_Soft_raw50
    if model_name == 'SoftFace_backbone':
        model_type = SoftFace_backbone
    if model_name == 'SoftFace_backboneraw':
        model_type = SoftFace_backboneraw
    if model_name == 'SoftFace_backbone_ResNet50raw':
        model_type = SoftFace_backbone_ResNet50raw
    if model_name == 'AdaTransLearn_Soft':
        model_type = AdaTransLearn_Soft
    if model_name == 'ElasticFaceD_backbone':
        model_type = ElasticFaceD_backbone
    if model_name == 'ElasticFace_OE':
        model_type = ElasticFace_OE
    if model_name == 'AttentionFace_backbone':
        model_type = AttentionFace_backbone
    if model_name == 'AdaTransLearn_ElasticAttention':
        model_type = AdaTransLearn_ElasticAttention
    if model_name == 'AdaTransLearn_ElasticD':
        model_type = AdaTransLearn_ElasticD
    if model_name == 'AdaTransLearn_ElasticD2':
        model_type = AdaTransLearn_ElasticD2
    if model_name == 'AdaTransLearn_ElasticD3':
        model_type = AdaTransLearn_ElasticD3
    print('Training mode: '+str(train_mode))

    train_dir = os.path.join(data_root, 'train.txt')
    val_dir = os.path.join(data_root, 'val.txt')
    test_dir = os.path.join(data_root, 'test.txt')
    test_large_dir = os.path.join(data_root, 'testlarge.txt')
    train_data_root = os.path.join(data_root, "train")
    test_data_root = os.path.join(data_root, "test")
    print(f'Current model type is {model_name}')
    print(f'Current model no is {model_no}')
    print(f'Current loss name is {loss_name}')
    print(f'Lock weights is {str(lock_weights)}\n')
    dataloaders, datasizes = load_dataset(train_data_root, test_data_root, train_dir, val_dir, test_dir, test_large_dir, batch_size, img_size, num_workers=num_workers)
    if train_mode:
        print(f'The training data resolution is {img_size}')
        print('................................')
        print('start training')
        print('................................')
        model= train(model_type, dataloaders, datasizes, regularizer_name, backbone_pre, model_name, model_no, learning_rate, margin=margin, scale=scale, dataroot=data_root, Temperature=temperature, T2=temperature2, lock_weights=lock_weights, loss_name=loss_name, loading_test=loading_test, args=args)
        
    else:
        print('loading model...')
        train_id = os.listdir(train_data_root)
        num_classes = len(train_id)
        model = load_model(model_class = model_type, model_name = model_name, model_no = model_no, num_class = num_classes)
        print('model loading finished...')
    
    #print(f'Find threshold is set to {find_th}')
    print(f'test flipped mode is {flip_test}')
    if hasattr(model, 'student'):
        model = model.student
    if model_name == 'SoftFace_backbone' or model_name == 'AdaTransLearn_ElasticAttention' or model_name == 'ElasticFace_OE':
        pass
    else:
        model = model.backbone
    
    

    #if find_th:
    #    sim_th = find_threshold(dataloaders['th'], model, flip_test)
    #print(f'the cosine similarity threshold for verification in training dataset is {sim_th}\n')
    print('testing process....')
    th = test_model(dataloaders['test'], model, flip_test, model_no, model_name)
    test_model_large(dataloaders['testlarge'], model, flip_test, th)
    #if sim_th > 0:
    #    print('test with threshold')
    #    test_model_withthreshold(dataloaders['test'], model, sim_th, flip_test, model_no, model_name)
    #else:
    #    print('test without threshold')
    #    test_model_withoutthreshold(dataloaders['test'], model, flip_test, model_no, model_name)
    print('process finished')
    print('\n'*2)

main()
