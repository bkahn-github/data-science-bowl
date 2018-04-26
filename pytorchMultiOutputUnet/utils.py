import os
import glob
import logging

import torch

from config import config

from sklearn.model_selection import KFold, train_test_split

class EarlyStopping:
    def __init__(self):
        
        self.best_score = 1e10
        self.best_epoch = 0 

    def evaluate(self, model, loss, epoch, patience=0):
        
        if loss < self.best_score:
            logging.info('Val score has improved, saving model')
            self.best_score = loss
            self.best_epoch = epoch
            return 'save'

        elif epoch - self.best_epoch > patience:
            logging.info('Val score hasn\'t improved for ' + str(epoch - self.best_epoch) + ' epochs, stopping training')
            return 'stop'
        else:
            logging.info('Val score hasn\'t improved for ' + str(epoch - self.best_epoch) + ' epochs, not saving model')
            return 'continue'
    
def calculate_losses(total_train_loss, total_val_loss, train_ids, val_ids, epoch):
    train_loss = total_train_loss / (len(train_ids) / config.BATCH_SIZE)
    val_loss = total_val_loss / (len(val_ids) / config.BATCH_SIZE)

    message = 'Epoch # ' + str(epoch) + ' | Training Loss: ' + str(round(train_loss, 4)) + ' | Validation Loss: ' + str(round(val_loss, 4))
    
    return message, train_loss, val_loss

def calculate_kfolds_losses(total_kfolds_train_loss, total_kfolds_val_loss, kfolds, epochs):
    train_loss = total_kfolds_train_loss / (kfolds * epochs)
    val_loss = total_kfolds_val_loss / (kfolds * epochs)

    message = '\nTotal loss over ' + str(kfolds) + ' kfolds and ' + str(epochs) + ' epochs | Training Loss: ' + str(round(train_loss, 4)) + ' | Validation Loss: ' + str(round(val_loss, 4))
    return message

def save_model(model, kfold):
    torch.save(model.state_dict(), './model-kfold-' + str(kfold) + '-best.pt')

def load_model(model, path):
    logging.info('Loading saved model')
    model.load_state_dict(torch.load(path))

    return model

def get_kfolds(kfolds):
    if config.SUBSET:
        ids = glob.glob(os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train', '*'))[:20]
    else:
        ids = glob.glob(os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train', '*'))

    ids = [id.split('/')[-1] for id in ids]

    if kfolds == 1:
        train_ids, val_ids = train_test_split(ids, test_size=0.1)
        return [[train_ids, val_ids]]
    else:
        kf = KFold(n_splits=kfolds)

        kfolds = []
        for x, y in kf.split(ids):
            x = ids[x[0]: x[-1]]
            y = ids[y[0]: y[-1]]

            kfolds.append([x, y])

        return kfolds

def get_path(id):
    img_path = os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train', id, 'images', id + '.png')

    mask_path = os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train_masks', id + '.png')
    contour_path = os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train_contours', id + '.png')
    center_path = os.path.join(config.ROOT_FOLDER, 'stage' + config.STAGE + '_train_centers', id + '.png')

    return img_path, mask_path, contour_path, center_path
