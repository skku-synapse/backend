import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import config as c
from model import get_cs_flow_model, save_model, FeatureExtractor, nf_forward
from utils import *


def train_api(train_loader, test_loader):
    model = get_cs_flow_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr_init, eps=1e-04, weight_decay=1e-5)
    model.to(c.device)
    if not c.pre_extracted:
        fe = FeatureExtractor()
        fe.eval()
        fe.to(c.device)
        for param in fe.parameters():
            param.requires_grad = False

    z_obs = Score_Observer('AUROC')

    train_start = int(round(time.time() * 1000)) # ms

    for epoch in range(c.meta_epochs):
        # train some epochs
        model.train()
        if c.verbose:
            print(F'\nTrain epoch {epoch}')
        for sub_epoch in range(c.sub_epochs):
            epoch_start = int(round(time.time() * 1000)) # ms
            train_loss = list()
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                optimizer.zero_grad()

                inputs, labels = preprocess_batch(data)  # move to device and reshape
                if not c.pre_extracted:
                    inputs = fe(inputs)

                z, jac = nf_forward(model, inputs)

                loss = get_loss(z, jac)
                train_loss.append(t2np(loss))

                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)
                optimizer.step()

            epoch_end = int(round(time.time() * 1000)) # ms
            mean_train_loss = np.mean(train_loss)
            if c.verbose:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f} \t time: {:d}ms'.format(epoch, sub_epoch, mean_train_loss, epoch_end-epoch_start))

        train_end = int(round(time.time() * 1000)) # ms

        print('\nTotal train time: {:d}ms\n'.format(train_end-train_start))

        # evaluate
        model.eval()
        if c.verbose:
            print('\nCompute loss and scores on test set:')
        test_loss = list()
        test_z = list()
        test_labels = list()

        predict_time = 0

        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
                inputs, labels = preprocess_batch(data)
                if not c.pre_extracted:
                    inputs = fe(inputs)
                    
                predict_start = int(round(time.time() * 1000)) # ms
                z, jac = nf_forward(model, inputs)
                z_concat = t2np(concat_maps(z))
                score = np.mean(z_concat ** 2, axis=(1, 2))
                predict_end = int(round(time.time() * 1000)) # ms
                predict_time += predict_end-predict_start
                loss = get_loss(z, jac)
                test_z.append(score)
                test_loss.append(t2np(loss))
                test_labels.append(t2np(labels))


        test_loss = np.mean(np.array(test_loss))
        if c.verbose:
            print('Epoch: {:d} \t test_loss: {:.4f} \t prediction time: {:d}ms'.format(epoch, test_loss, predict_time))

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        anomaly_score = np.concatenate(test_z, axis=0)
        z_obs.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
                     print_score=c.verbose or epoch == c.meta_epochs - 1)

    if c.save_model:
        model.to('cpu')
        save_model(model, c.modelname)

    return z_obs.max_score, z_obs.last, z_obs.min_loss_score
