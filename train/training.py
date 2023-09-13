import matplotlib.pyplot as plt
import time
import copy
import torch
import math
import numpy as np
from utils.helper_functions import *
from torch.nn.utils import clip_grad_norm_
from datetime import timedelta

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, model_name, runs=0, lock_weights=False, test=False):
    since = time.time()
    batch_start_time = since
    batch_end_time = since
    end_time = 216000
    print(f'will terminate at {str(timedelta(seconds=end_time))}')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = math.inf
    early_stopper = EarlyStopper(patience=10, min_delta=5e-3)
    early_stopping = False
    ######## Draw loss and accuracy ##############
    y_loss = {}  # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []

    x_epoch = []
    #fig = plt.figure()
    #ax0 = fig.add_subplot(121, title="loss")
    #ax1 = fig.add_subplot(122, title="top1err")


    def draw_curve(current_epoch):
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['train'], 'b', label='train')
        ax0.plot(x_epoch, y_loss['val'], 'r', label='val')
        ax1.plot(x_epoch, y_err['train'], 'b', label='train')
        ax1.plot(x_epoch, y_err['val'], 'r', label='val')
        if current_epoch == 0:
            ax0.legend()
            ax1.legend()
        fig.savefig('/project/6003167/zzh2015/plots'+model_name+str(runs)+'.png')
    ###############################################################

    ############## Training process ############################
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        data_reading_time = 0
        data_training_time = 0
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            kd_loss = 0.0
            ce_loss = 0.0
            #running_corrects = 0

            # Iterate over data.
            for inputs in dataloaders[phase]:
                batch_start_time = time.time()
                #if test:
                    #print(f'batch data loading time: {str(timedelta(seconds=batch_start_time-batch_end_time))}')
                data_reading_time += batch_start_time-batch_end_time
                gpu_inputs = dict_to_device(inputs, 'cuda')
                #inputs = inputs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                if lock_weights:
                    model.set_backbone_trainable(False)
                with torch.set_grad_enabled(phase == 'train'):
                    #with torch.autograd.detect_anomaly():
                        out = model(gpu_inputs['image'], gpu_inputs['label'])
                        loss = criterion(out, gpu_inputs['label'], gpu_inputs['age'])
                        if type(loss) is tuple:
                            kd_loss += loss[1].item() * inputs['image'].size(0)
                            ce_loss += loss[2].item() * inputs['image'].size(0)
                            loss = loss[0]

                    # backward + optimize only if in training phase
                        if phase == 'train':
                            if lock_weights:
                                if model.check_trainable():
                                    print('weights unlocked but requires gradient')
                                    return
                            loss.backward()
                            if hasattr(model, 'student'):
                                clip_grad_norm_(model.student.parameters(), max_norm=5, norm_type=2)
                            else:
                                clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
                            optimizer.step()

                # statistics
                running_loss += loss.item() * inputs['image'].size(0)
                batch_end_time = time.time()
                #if test:
                    #print(f'batch training time: {str(timedelta(seconds=batch_end_time-batch_start_time))}')
                data_training_time += batch_end_time-batch_start_time
                #running_corrects += torch.sum(preds == labels.data)
            #if phase == 'train':
            #    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_kd_loss = kd_loss / dataset_sizes[phase]
            epoch_ce_loss = ce_loss / dataset_sizes[phase]
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]
            y_loss[phase].append(epoch_loss)
            #y_err[phase].append(torch.Tensor.cpu(epoch_acc))

            print(f'{phase} Loss: {epoch_loss:.4f}')
            if kd_loss != 0.0:
                print(f'{phase} first_Loss: {epoch_kd_loss:.4f}')
            if ce_loss != 0.0:
                print(f'{phase} second_Loss: {epoch_ce_loss:.4f}')
            if phase == 'val':
                scheduler.step(epoch_loss)
                # draw loss and acc curve
                #draw_curve(epoch)

                # deep copy the model
                if epoch_loss<best_loss:
                  best_loss = epoch_loss
                  best_model_wts = copy.deepcopy(model.state_dict())
                early_stopping = early_stopper.early_stop(epoch_loss)
        elapsed = time.time() - since
        print(f'Have trained {str(timedelta(seconds=elapsed))}')
        print(f'Data reading time: {str(data_reading_time)}')
        print(f'Data training time: {str(data_training_time)}')
        if elapsed > end_time:
            print(f'Over {str(timedelta(seconds=end_time))}, stop early')
            break
        if early_stopping:
          print('Early stopping.')
          break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model