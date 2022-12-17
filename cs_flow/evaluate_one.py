from copy import deepcopy
import time
import numpy as np
import torch
from cs_flow.model import load_model, FeatureExtractor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import PIL
from os.path import join
import os
from PIL import Image
from torchvision import transforms

CLASS_NAME = "camera_cropped"
MODEL_NAME = "camera_cropped"
IMG_SIZE = (512, 512)
DEVICE = 'cuda'
NORM_MEAN, NORM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
PRE_EXTRACTED = False
EXTRACTOR = "effnetB5"
N_FEAT = {"effnetB5": 304}[EXTRACTOR]
THRESHOLD = 0.6938925385475159

localize = True
upscale_mode = 'bilinear'
score_export_dir = '/home/synapse/simulator/backend/static/histogram'

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def flat(tensor):
    return tensor.reshape(tensor.shape[0], -1)


def concat_maps(maps):
    flat_maps = list()
    for m in maps:
        flat_maps.append(flat(m))
    return torch.cat(flat_maps, dim=1)[..., None]

# localization을 위한 이미지 생성 함수
def viz_maps(img_path, z, name, model_name):
    map_export_dir = join('/home/synapse/simulator/backend/static/test_map', "")
    os.makedirs(map_export_dir, exist_ok=True)

    image = PIL.Image.open(img_path).convert('RGB')
    image = np.array(image)

    z_grouped = list()
    likelihood_grouped = list()
    all_maps = list()
    for i in range(len(z)):
        z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
        likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)) / N_FEAT)    

    all_maps.extend(likelihood_grouped[0])
    map_to_viz = t2np(F.interpolate(all_maps[0][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
        0, 0]

    # 원본 이미지, 히트맵 이미지, 원본+히트맵 overlay 이미지를 각각 저장
    plt.clf()
    plt.imshow(map_to_viz)
    plt.axis('off')
    plt.savefig(join(map_export_dir, name + '_map.jpg'), bbox_inches='tight', pad_inches=0)

    plt.clf()
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(join(map_export_dir, name + '_orig.jpg'), bbox_inches='tight', pad_inches=0)
    plt.imshow(map_to_viz, cmap='viridis', alpha=0.7)
    plt.savefig(join(map_export_dir, 'overlay.jpg'), bbox_inches='tight', pad_inches=0)
    return

# score histogram 생성 함수
# thresh : 최대 score 값 ( x축 )
def compare_histogram(scores, classes, dataset, thresh=7, n_bins=32):
    classes = deepcopy(classes)
    scores = deepcopy(scores)
    scores = [thresh if x > thresh else x for x in scores]
    bins = np.linspace(np.min(scores), np.max(scores), n_bins)

    scores_norm = []
    scores_ano = []
    for i in range(0, len(scores)):
        if classes[i] == 1:
            scores_ano.append(scores[i])
        else:
            scores_norm.append(scores[i])

    print(scores_norm, scores_ano)

    plt.clf()
    plt.figure().set_facecolor("#1e1e1e")
    ax = plt.axes()
    ax.set_facecolor("#1e1e1e")
    ax.tick_params(axis='x', colors='white', labelsize=15)
    ax.tick_params(axis='y', colors='white', labelsize=15)
    
    plt.hist(scores_norm, bins, alpha=0.5, label='non-defects', color='cyan', edgecolor="#1e1e1e")
    plt.hist(scores_ano, bins, alpha=0.5, label='defects', color='crimson', edgecolor="#1e1e1e")

    ticks = np.linspace(0.5, thresh, 5)
    labels = [str(i) for i in ticks[:-1]] + ['>' + str(thresh)]
    plt.xticks(ticks, labels=labels)
    plt.xlabel('score', fontsize=18)
    plt.ylabel('Count (normalized)', fontsize=18)
    plt.legend()
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    plt.grid(axis='y')
    
    plt.savefig(join(score_export_dir, dataset + '.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

# 하나의 이미지 양/불량 판단 함수
def evaluate_one(model, img_path, model_name="SMT", threshold=1.192936):
    model.to(DEVICE)
    model.eval()

    if not PRE_EXTRACTED:
        fe = FeatureExtractor()
        fe.eval()
        fe.to(DEVICE)
        for param in fe.parameters():
            param.requires_grad = False

    
    img = Image.open(img_path)

    start = time.time()
    tfs = [transforms.Resize(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)]
    transform = transforms.Compose(tfs)

    img = transform(img)

    # preprocess_batch
    '''move data to device and reshape image'''
    img = img.to(DEVICE)
    img = img.view(-1, *img.shape[-3:])

    if not PRE_EXTRACTED:
        img = fe(img)
    
    z = model(img)

    z_concat = t2np(concat_maps(z))
    nll_score = np.std(z_concat ** 2 / 2, axis=(1, 2))

    end = time.time()
    
    viz_maps(img_path, z, "test_img", model_name)

    # anomaly score, 양/불량 여부, 예측 시간을 반환
    result = {
        "anomaly_score" : float(nll_score[0]),
        "isAnomaly" : bool(nll_score[0] > threshold),
        "time": float(end-start)*1000
    }

    return result


def evaluate_function(img_path, model_name="SMT", threshold=1.192936):

    # 데이터셋에 따른 모델 로드
    mod = load_model(model_name)

    # default threshold: SMT 기준
    # dataset에 따라 threshold 지정 후 evaluate

    if model_name == "lens":
        threshold = 3.077919
    elif model_name == "flex":
        threshold = 1.5589128
    return evaluate_one(mod, img_path, model_name, threshold)