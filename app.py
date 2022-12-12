import base64
import sys
import os
import time
import math
import json
import random

from flask import Flask, make_response, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import torch
#from backend.cs_flow.evaluate_one import evaluate_function

import timm
from timm.data import resolve_data_config
#from cs_flow import *
from cs_flow.evaluate_one import evaluate_function, compare_histogram
# from cs_flow.train_api import train_api

from setproctitle import *

#from fastflow.synapse_main import evaluate_image, parse_args

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# setproctitle('synapse-backend')

api = Flask(__name__)
CORS(api, support_credentials=True)

CS_MODEL_DIR = 'cs_flow/models/'

global scores
global classes

scores = []
classes = []

@api.route('/getDataList', methods=['GET'])
def getDataList():
    dataset = request.args.get('dataset')
    root_dir = '/home/synapse/simulator/backend/static/' + dataset  # 디렉토리

    ok_list = []
    ng_list = []
    possible_img_extension = ['.jpg', '.jpeg',
                              '.JPG', '.bmp', '.png']  # 이미지 확장자들

    for (root, dirs, files) in os.walk(root_dir):
        if len(files) > 0:
            for file_name in files:
                if os.path.splitext(file_name)[1] in possible_img_extension:
                    #img_path = root + '/' + file_name
                    img_path = file_name
                    # 경로에서 \를 모두 /로 바꿔줘야함
                    img_path = img_path.replace('\\', '/')  # \는 \\로 나타내야함
                    if (root[-2:] == "OK"):
                        ok_list.append("OK/"+img_path)
                    else:
                        ng_list.append("NG/"+img_path)

    cs_path = os.path.join(CS_MODEL_DIR, "camera_lens")
    cs_model = torch.load(cs_path)

    return {"ok": ok_list, "ng": ng_list}


@api.route('/getAllData', methods=['GET'])
def getAllData():
    global scores
    global classes

    scores = []
    classes = []

    dataset = request.args.get('dataset')
    root_dir = '/home/synapse/simulator/backend/static/' + dataset  # 디렉토리

    data_list = []
    possible_img_extension = ['.jpg', '.jpeg',
                              '.JPG', '.bmp', '.png']  # 이미지 확장자들

    for (root, dirs, files) in os.walk(root_dir):
        if len(files) > 0:
            for file_name in files:
                if os.path.splitext(file_name)[1] in possible_img_extension:
                    #img_path = root + '/' + file_name
                    img_path = file_name
                    # 경로에서 \를 모두 /로 바꿔줘야함
                    img_path = img_path.replace('\\', '/')  # \는 \\로 나타내야함
                    if (root[-2:] == "OK"):
                        data_list.append("OK/"+img_path)
                    else:
                        data_list.append("NG/"+img_path)

    cs_path = os.path.join(CS_MODEL_DIR, "camera_lens")
    cs_model = torch.load(cs_path)
    random.shuffle(data_list)

    

    return {"list": data_list}


@api.route('/getImage', methods=['GET'])
def getImage():
    dataset = request.args.get('dataset')
   
    user_img_path = '/home/synapse/simulator/backend/static/histogram/' + dataset + ".png"
    with open(user_img_path, "rb") as f:
        image_binary = f.read()

        response = make_response(base64.b64encode(image_binary))
        response.headers.set('Content-Type', 'image/jpg')
        response.headers.set('Content-Disposition',
                             'attachment', filename='image.jpg')
        image = base64.b64encode(image_binary).decode("utf-8")
        return jsonify({'status': True, 'image': image})


@api.route('/getHistogram', methods=['GET'])
def getHistogram():

    dataset = request.args.get('dataset')

    compare_histogram(scores, classes, dataset)

    user_img_path = '/home/synapse/simulator/backend/static/histogram/' + dataset + ".png"
    with open(user_img_path, "rb") as f:
        image_binary = f.read()

        response = make_response(base64.b64encode(image_binary))
        response.headers.set('Content-Type', 'image/jpg')
        response.headers.set('Content-Disposition',
                             'attachment', filename='image.jpg')
        image = base64.b64encode(image_binary).decode("utf-8")
        return jsonify({'status': True, 'image': image})

    return jsonify({'success': True})


@api.route('/predict', methods=['GET'])
def predict():
    print('start!')
    model = request.args.get('model')
    dataset = request.args.get('dataset')  # module / lens / flex
    img_name = request.args.get('img_name')
    print(dataset)
    user_img_path = '/home/synapse/simulator/backend/static/' + dataset + "/" + img_name

    if model == "CS-Flow":
        if dataset == "module":
            response = evaluate_function(user_img_path)
        elif dataset == "lens":
            response = evaluate_function(user_img_path, "camera_lens")
        elif dataset == "flex":
            response = evaluate_function(user_img_path, "board")
        else:
            response = {
                'image': user_img_path,
                'isAnomaly': False,
            }

        scores.append(response["anomaly_score"])
        
        if img_name[0] == "O":
            response["label"] = "OK"
            classes.append(0)
        else:
            response["label"] = "NG"
            classes.append(1)

        with open(user_img_path, "rb") as f:
            image_binary = f.read()

            img_response = make_response(base64.b64encode(image_binary))
            img_response.headers.set('Content-Type', 'image/jpg')
            img_response.headers.set(
                'Content-Disposition', 'attachment', filename='image.jpg')
            image = base64.b64encode(image_binary).decode("utf-8")
            response["image"] = image

        hitmap_path = './static/test_map/overlay.jpg'
        with open(hitmap_path, "rb") as f:
            image_binary = f.read()

            img_response = make_response(base64.b64encode(image_binary))
            img_response.headers.set('Content-Type', 'image/jpg')
            img_response.headers.set(
                'Content-Disposition', 'attachment', filename='image.jpg')
            image = base64.b64encode(image_binary).decode("utf-8")
            response["overlay"] = image
        # return jsonify({'status': True, 'image': image})

        return json.dumps(response)

# @api.route('/train', methods=['POST'])
# def train():
#     print('train setting start!')
#     model_name = request.args.get('model_name')
#     dataset_name = request.args.get('dataset_name')
#     dataset_path = request.args.get('dataset_path')    
#     metaEpoch = request.args.get('metaEpoch')
#     subEpoch = request.args.get('subEpoch')
#     batchSize = request.args.get('batchSize')
#     extractor = request.args.get('extractor')
#     learningRate = request.args.get('learningRate')

#     #딕셔너리로 만들어서 인자로 보내주기
#     info = dict()

#     info['model_name'] = model_name #모델 이름
#     info['dataset_name'] = dataset_name #데이터셋 이름
#     info['dataset_path'] = dataset_path #데이터셋 경로
#     info['metaEpoch'] = int(metaEpoch) 
#     info['subEpoch'] = int(subEpoch)
#     info['batchSize'] = int(batchSize)
#     info['extractor'] = extractor #feature extractor
#     info['learningRate'] = float(learningRate)

#     train_api(info)

#     print('trian start!')


api.run(host='0.0.0.0', port=51122, debug=True)
