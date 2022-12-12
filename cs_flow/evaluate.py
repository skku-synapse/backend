import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm
from model import load_model, FeatureExtractor
import config as c
from utils import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
import PIL
from os.path import join
import os
from copy import deepcopy

from setproctitle import *

setproctitle('cs-flow')

localize = True
upscale_mode = 'bilinear'
score_export_dir = join('./viz/scores/', c.modelname)
os.makedirs(score_export_dir, exist_ok=True)
map_export_dir = join('./viz/maps/', c.modelname)
os.makedirs(map_export_dir, exist_ok=True)


def compare_histogram(scores, classes, thresh=2.5, n_bins=64):
    classes = deepcopy(classes)
    classes[classes > 0] = 1
    scores[scores > thresh] = thresh
    bins = np.linspace(np.min(scores), np.max(scores), n_bins)
    scores_norm = scores[classes == 0]
    scores_ano = scores[classes == 1]

    plt.clf()
    plt.hist(scores_norm, bins, alpha=0.5, density=True, label='non-defects', color='cyan', edgecolor="black")
    plt.hist(scores_ano, bins, alpha=0.5, density=True, label='defects', color='crimson', edgecolor="black")

    ticks = np.linspace(0.5, thresh, 5)
    labels = [str(i) for i in ticks[:-1]] + ['>' + str(thresh)]
    plt.xticks(ticks, labels=labels)
    plt.xlabel(r'$-log(p(z))$')
    plt.ylabel('Count (normalized)')
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(join(score_export_dir, 'score_histogram.png'), bbox_inches='tight', pad_inches=0)


def viz_roc(values, classes, class_names):
    def export_roc(values, classes, export_name='all'):
        # Compute ROC curve and ROC area for each class
        classes = deepcopy(classes)
        classes[classes > 0] = 1
        fpr, tpr, _ = roc_curve(classes, values)
        roc_auc = auc(fpr, tpr)

        plt.clf()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for class ' + c.class_name)
        plt.legend(loc="lower right")
        plt.axis('equal')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.savefig(join(score_export_dir, export_name + '.png'))

    export_roc(values, classes)
    for cl in range(1, classes.max() + 1):
        filtered_indices = np.concatenate([np.where(classes == 0)[0], np.where(classes == cl)[0]])
        classes_filtered = classes[filtered_indices]
        values_filtered = values[filtered_indices]
        export_roc(values_filtered, classes_filtered, export_name=class_names[filtered_indices[-1]])


def viz_maps(maps, name, label):
    img_path = img_paths[c.viz_sample_count]
    image = PIL.Image.open(img_path).convert('RGB')
    image = np.array(image)

    map_to_viz = t2np(F.interpolate(maps[0][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
        0, 0]

    plt.clf()
    plt.imshow(map_to_viz)
    plt.axis('off')
    plt.savefig(join(map_export_dir, name + '_map.jpg'), bbox_inches='tight', pad_inches=0)

    if label > 0:
        plt.clf()
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(join(map_export_dir, name + '_orig.jpg'), bbox_inches='tight', pad_inches=0)
        plt.imshow(map_to_viz, cmap='viridis', alpha=0.3)
        plt.savefig(join(map_export_dir, name + '_overlay.jpg'), bbox_inches='tight', pad_inches=0)
    return


def viz_map_array(maps, labels, n_col=8, subsample=4, max_figures=-1):
    plt.clf()
    fig, subplots = plt.subplots(3, n_col)

    fig_count = -1
    col_count = -1
    for i in range(len(maps)):
        if i % subsample != 0:
            continue

        if labels[i] == 0:
            continue

        col_count = (col_count + 1) % n_col
        if col_count == 0:
            if fig_count >= 0:
                plt.savefig(join(map_export_dir, str(fig_count) + '.jpg'), bbox_inches='tight', pad_inches=0)
                plt.close()
            fig, subplots = plt.subplots(3, n_col, figsize=(22, 8))
            fig_count += 1
            if fig_count == max_figures:
                return

        anomaly_description = img_paths[i].split('/')[-2]
        image = PIL.Image.open(img_paths[i]).convert('RGB')
        image = np.array(image)
        map = t2np(F.interpolate(maps[i][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
            0, 0]
        subplots[1][col_count].imshow(map)
        subplots[1][col_count].axis('off')
        subplots[0][col_count].imshow(image)
        subplots[0][col_count].axis('off')
        subplots[0][col_count].set_title(c.class_name + ":\n" + anomaly_description)
        subplots[2][col_count].imshow(image)
        subplots[2][col_count].axis('off')
        subplots[2][col_count].imshow(map, cmap='viridis', alpha=0.3)
    for i in range(col_count, n_col):
        subplots[0][i].axis('off')
        subplots[1][i].axis('off')
        subplots[2][i].axis('off')
    if col_count > 0:
        plt.savefig(join(map_export_dir, str(fig_count) + '.jpg'), bbox_inches='tight', pad_inches=0)
    return

def viz_map_array_error(maps, labels, fps, fns, anomaly_score, n_col=8, max_figures=-1):
        
    plt.clf()
    fig, subplots = plt.subplots(3, n_col)

    fig_count = -1
    col_count = -1
    for i in range(len(maps)):
        
        img_name = img_paths[i].split('/')[-1]
        if img_name not in fps and img_name not in fns:
            continue

        col_count = (col_count + 1) % n_col
        if col_count == 0:
            if fig_count >= 0:
                plt.savefig(join(map_export_dir, str(fig_count) + '.jpg'), bbox_inches='tight', pad_inches=0)
                plt.close()
            fig, subplots = plt.subplots(3, n_col, figsize=(22, 8))
            fig_count += 1
            if fig_count == max_figures:
                return

        anomaly_description = img_paths[i].split('/')[-2]
        image = PIL.Image.open(img_paths[i]).convert('RGB')
        image = np.array(image)
        
        #print(image)
        map = t2np(F.interpolate(maps[i][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
            0, 0]
        subplots[1][col_count].imshow(map)
        subplots[1][col_count].axis('off')
        subplots[0][col_count].imshow(image)
        subplots[0][col_count].axis('off')
        subplots[0][col_count].set_title(c.class_name + ":\n" + anomaly_description + "\nscore: " + str(anomaly_score[i]))
        subplots[2][col_count].imshow(image)
        subplots[2][col_count].axis('off')
        subplots[2][col_count].imshow(map, cmap='viridis', alpha=0.3)
    for i in range(col_count, n_col):
        subplots[0][i].axis('off')
        subplots[1][i].axis('off')
        subplots[2][i].axis('off')
    if col_count > 0:
        plt.savefig(join(map_export_dir, str(fig_count) + '.jpg'), bbox_inches='tight', pad_inches=0)
    return


def evaluate(model, test_loader):
    model.to(c.device)
    model.eval()
   
    if not c.pre_extracted:
        fe = FeatureExtractor()
        fe.eval()
        fe.to(c.device)
        for param in fe.parameters():
            param.requires_grad = False

    print('\nCompute maps, loss and scores on test set:')
    evaluate_start = int(round(time.time() * 1000)) # ms
    anomaly_score = list()
    test_labels = list()
    c.viz_sample_count = 0
    all_maps = list()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            inputs, labels = preprocess_batch(data)
            # print(labels) ok: 0, ng: 1
            if not c.pre_extracted:
                inputs = fe(inputs)
            z = model(inputs)
            
            z_concat = t2np(concat_maps(z))
            nll_score = np.std(z_concat ** 2 / 2, axis=(1, 2))
            anomaly_score.append(nll_score)
            test_labels.append(t2np(labels))

            if localize:
                z_grouped = list()
                likelihood_grouped = list()
                for i in range(len(z)):
                    #print(len(z))
                    z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
                    likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)) / c.n_feat)
                all_maps.extend(likelihood_grouped[0])
                for i_l, l in enumerate(t2np(labels)):
                    # viz_maps([lg[i_l] for lg in likelihood_grouped], c.modelname + '_' + str(c.viz_sample_count), label=l, show_scales = 1)
                    c.viz_sample_count += 1
    evaluate_end = int(round(time.time() * 1000)) # ms
    print('\nTotal evaluate time: {:d}ms\n'.format(evaluate_end-evaluate_start))
    anomaly_score = np.concatenate(anomaly_score)
    test_labels = np.concatenate(test_labels)

    compare_histogram(anomaly_score, test_labels)

    class_names = [img_path.split('/')[-2] for img_path in img_paths]
    viz_roc(anomaly_score, test_labels, class_names)

    test_labels = np.array([1 if l > 0 else 0 for l in test_labels])
    auc_score = roc_auc_score(test_labels, anomaly_score)
    fpr, tpr, thresholds = roc_curve(test_labels, anomaly_score)

    J = tpr - fpr
    # gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(J)
    best_threshold = thresholds[ix]

    print('Best Threshold {}, tpr {}, fpr {}'.format(best_threshold, tpr[ix], fpr[ix]))
    print('AUC:', auc_score)

    false_positives = []
    false_negatives = []

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    count = 1

    for i in range(len(test_labels)):
        # False negative
        if test_labels[i] == 0 and anomaly_score[i] > best_threshold: 
            print(count, "실제 ok, 예측 ng", img_paths[i].split('/')[-1])
            count += 1
            false_negatives.append(img_paths[i].split('/')[-1])
            FN += 1
        # True positive
        elif test_labels[i] == 0: 
            TP += 1
        # False positive
        elif test_labels[i] == 1 and anomaly_score[i] < best_threshold: 
            print(count, "실제 ng, 예측 ok", img_paths[i].split('/')[-1])
            false_positives.append(img_paths[i].split('/')[-1])
            count += 1
            FP += 1
        # True negative
        else: 
            TN += 1

    print("precision: {}\trecall: {}\taccuracy: {}\tfalse alarm: {}".format(TP/(TP+FP), TP/(TP+FN), (TP+TN)/(TP+FN+FP+TN), FP/(FP+TN)))

    if localize:
        # viz_map_array(all_maps, test_labels)
        viz_map_array_error(all_maps, test_labels, false_positives, false_negatives, anomaly_score)

    return

train_set, test_set = load_datasets(c.dataset_path, c.class_name)
img_paths = test_set.paths if c.pre_extracted else [p for p, l in test_set.samples]
_, test_loader = make_dataloaders(train_set, test_set)
mod = load_model(c.modelname)
evaluate(mod, test_loader)
