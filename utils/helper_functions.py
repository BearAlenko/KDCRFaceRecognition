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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
#inputs = next(iter(dataloaders['train']))

# Make a grid from batch
#out = torchvision.utils.make_grid(inputs['image'])

#imshow(out, title=inputs['label'])

def dict_to_device(orig, device):
    new = {}
    for k,v in orig.items():
        new[k] = v.to(device)
    return new

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def save_model(model, file_path = '/project/6003167/zzh2015/trained_model/saved_', model_name = 'AICNN', model_no = 1, file_format='.pt'):
  saved_model_path = file_path+model_name+'_'+str(model_no)+file_format
  torch.save(model.state_dict(), saved_model_path)

def load_model(model_class, file_path = '/project/6003167/zzh2015/trained_model/saved_', model_name = 'AICNN', model_no = 1, file_format='.pt', num_class=3015):
  saved_model_path = file_path+model_name+'_'+str(model_no)+file_format
  model3 = model_class(num_classes = num_class)
  model3.to('cuda')
  model3.load_state_dict(torch.load(saved_model_path))
  return model3

def test_model_f(testloader, model, similarity_threshold):
    model.eval()
    best_acc = 0
    correct = 0
    n = 0
    sims = []
    labels_true = []
    with torch.no_grad():
        for inputs in testloader:
            image0 = inputs['image0'].to(device)
            image1 = inputs['image1'].to(device)

            id_out0 = model(image0)
            id_out1 = model(image1)
            cal_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
            result = cal_similarity(id_out0, id_out1)
            #sims += result.detach().cpu()
            #labels_true += inputs['label']
            same_identity = result > similarity_threshold
            same_identity = same_identity.long().cpu()
            n += len(inputs['label'])
            correct += (same_identity == inputs['label']).sum().item()
    #for i in range(len(sims)):
    #    if sims[i] < 0:
    #        continue
    #    threshold = sims[i]
    #    test_correct = 0
    #    for sim in sims:
    #        if (sim >= threshold)==labels_true[i]:
    #            test_correct += 1
    #    acc = test_correct/len(sims)
    #    if acc > best_acc:
    #        best_acc = acc
    #        best_th = threshold
    #        print(f'current best acc is {best_acc}, current best threshold is {best_th}')

    print(f'Accuracy of the network on the {n} test images: {100 * correct/n} % and the threshold is {similarity_threshold}')

def test_model_withoutthreshold(testloader, model, flip, model_no, model_name):
    model.eval()
    best_acc = 0
    best_th = 0
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
            if flip:
                flipped_image0 = torch.flip(image0, dims=(2,))
                flipped_image1 = torch.flip(image1, dims=(2,))
            
                id_flipout0 = model(flipped_image0)
                id_flipout1 = model(flipped_image1)
                id_out0 = torch.cat((id_out0, id_flipout0), 1)
                id_out1 = torch.cat((id_out1, id_flipout1), 1)
            id_out0 = l2_norm(id_out0, axis=1)
            id_out1 = l2_norm(id_out1, axis=1)
            cal_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
            result = cal_similarity(id_out0, id_out1)
            sims += result.detach().cpu()
            labels_true += inputs['label']
            #same_identity = result > similarity_threshold
            #same_identity = same_identity.long().cpu()
            #n += len(inputs['label'])
            #correct += (same_identity == inputs['label']).sum().item()
    for th in sims:
        threshold = th
        test_correct = 0
        for i, sim in enumerate(sims):
            if (sim >= threshold)==labels_true[i]:
                test_correct += 1
        acc = test_correct/len(sims)
        if acc > best_acc:
            best_acc = acc
            best_th = threshold
            print(f'current best acc = {best_acc}, th = {best_th}')
    df = pd.DataFrame(list(zip([sim.item() for sim in sims], [label.item() for label in labels_true])),
               columns =['sims', 'labels'])
    today = str(date.today())
    filepath = os.path.join('/project/6003167/zzh2015/trained_model/test_result', model_name+'_'+str(model_no)+'_'+str(flip)+'_'+today+'.csv')  
    #filepath.parent.mkdir(parents=True, exist_ok=True) 
    df = df.to_csv(filepath, index=False)
    print(f'{len(sims)} test images: acc = {100 * best_acc} %, th = {best_th}')


def test_model_withthreshold(testloader, model, similarity_threshold, flip, model_no, model_name):
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
            if flip:
                flipped_image0 = torch.flip(image0, dims=(2,))
                flipped_image1 = torch.flip(image1, dims=(2,))
                id_flipout0 = model(flipped_image0)
                id_flipout1 = model(flipped_image1)
                id_out0 = torch.cat((id_out0, id_flipout0), 1)
                id_out1 = torch.cat((id_out1, id_flipout1), 1)
            
            id_out0 = l2_norm(id_out0, axis=1)
            id_out1 = l2_norm(id_out1, axis=1)
            cal_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
            result = cal_similarity(id_out0, id_out1)
            sims += result.detach().cpu()
            labels_true += inputs['label']
    print(f'the number of calculated similarity pairs is {len(sims)}')
    test_correct = 0
    tp = 0
    tn = 0
    for i, sim in enumerate(sims):
     
        if (sim >= similarity_threshold)==labels_true[i]:
            test_correct += 1
            if labels_true[i] == 1:
                tp += 1
            else:
                tn += 1
    tp_r = tp/(len(sims)/2)
    tn_r = tn/(len(sims)/2)
    acc = test_correct/len(sims)
    print(f'true positive rate = {tp_r*100} %, true negative rate = {tn_r*100} %')
    print(f'Accuracy of the network on the {len(sims)} test images: {100 * acc} % and the threshold is {similarity_threshold}')
    df = pd.DataFrame(list(zip([sim.item() for sim in sims], [label.item() for label in labels_true])),
               columns =['sims', 'labels'])
    today = str(date.today())
    filepath = os.path.join('/project/6003167/zzh2015/trained_model/test_result', model_name+'_'+str(model_no)+'_'+str(flip)+'_'+today+'.csv')  
    #filepath.parent.mkdir(parents=True, exist_ok=True) 
    df = df.to_csv(filepath, index=False)