import base64
import sys
import os
import time
import math
import json
import random
import csv

from flask import Flask, make_response, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import torch

from cs_flow.evaluate_one import evaluate_function, compare_histogram

api = Flask(__name__)
CORS(api, support_credentials=True)

# 모델 디렉토리
CS_MODEL_DIR = 'cs_flow/models/'

global scores
global classes

scores = []
classes = []

# 모든 데이터 리스트 반환 함수
@api.route('/getAllData', methods=['GET'])
def getAllData():
    global scores
    global classes

    scores = []
    classes = []

    # argument(데이터셋 종류)
    dataset = request.args.get('dataset')

    # 결과 저장 파일 생성
    f = open(dataset + '_result.csv', 'w')
    wr = csv.writer(f)
    wr.writerow(["file name", "label", "prediction", "anomaly score"])
    f.close()

    # 데이터셋 디렉토리
    root_dir = '/home/synapse/simulator/backend/static/' + dataset  

    data_list = []
    possible_img_extension = ['.jpg', '.jpeg',
                              '.JPG', '.bmp', '.png']  # 이미지 확장자들

    for (root, dirs, files) in os.walk(root_dir):
        if len(files) > 0:
            for file_name in files:
                if os.path.splitext(file_name)[1] in possible_img_extension:
                    img_path = file_name
                    # 경로에서 \를 /로 대체
                    img_path = img_path.replace('\\', '/')
                    if (root[-2:] == "OK"):
                        data_list.append("OK/"+img_path)
                    else:
                        data_list.append("NG/"+img_path)

    # 모든 이미지 랜덤으로 순서 변경
    random.shuffle(data_list)

    return {"list": data_list}

# score histogram 반환
@api.route('/getHistogram', methods=['GET'])
def getHistogram():

    # argument(데이터셋 종류)
    dataset = request.args.get('dataset')

    compare_histogram(scores, classes, dataset)

    # 히스토그램 이미지 경로
    user_img_path = '/home/synapse/simulator/backend/static/histogram/' + dataset + ".png"

    # 이미지 반환
    with open(user_img_path, "rb") as f:
        image_binary = f.read()

        response = make_response(base64.b64encode(image_binary))
        response.headers.set('Content-Type', 'image/jpg')
        response.headers.set('Content-Disposition',
                             'attachment', filename='image.jpg')
        image = base64.b64encode(image_binary).decode("utf-8")
        return jsonify({'status': True, 'image': image})

    return jsonify({'success': True})

# 이미지 양/불량 판단
@api.route('/predict', methods=['GET'])
def predict():

    # arguments(모델 이름(CS-Flow), 데이터셋 종류, 이미지 이름)
    model = request.args.get('model')
    dataset = request.args.get('dataset')  # module / lens / flex
    img_name = request.args.get('img_name')

    # 예측 결과 저장 파일 로드
    f = open(dataset + '_result.csv', 'a')
    wr = csv.writer(f)

    # 이미지 경로
    user_img_path = '/home/synapse/simulator/backend/static/' + dataset + "/" + img_name

    if model == "CS-Flow":
        if dataset == "module":
            response = evaluate_function(user_img_path)
        elif dataset == "lens":
            response = evaluate_function(user_img_path, "lens")
        elif dataset == "flex":
            response = evaluate_function(user_img_path, "flex")
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

        # 예측 결과를 파일에 저장
        if response['isAnomaly'] is True:
            wr.writerow([img_name[3:], response["label"], "NG", response["anomaly_score"]])
        else:
            wr.writerow([img_name[3:], response["label"], "OK", response["anomaly_score"]])
        f.close()

        # 원본 이미지 인코딩
        with open(user_img_path, "rb") as f:
            image_binary = f.read()

            img_response = make_response(base64.b64encode(image_binary))
            img_response.headers.set('Content-Type', 'image/jpg')
            img_response.headers.set(
                'Content-Disposition', 'attachment', filename='image.jpg')
            image = base64.b64encode(image_binary).decode("utf-8")
            response["image"] = image

        # overlay 이미지 인코딩
        hitmap_path = './static/test_map/overlay.jpg'
        with open(hitmap_path, "rb") as f:
            image_binary = f.read()

            img_response = make_response(base64.b64encode(image_binary))
            img_response.headers.set('Content-Type', 'image/jpg')
            img_response.headers.set(
                'Content-Disposition', 'attachment', filename='image.jpg')
            image = base64.b64encode(image_binary).decode("utf-8")
            response["overlay"] = image

        # 이미지 기존 label, 예측 결과, 원본 이미지, overlay 이미지 반환
        return json.dumps(response)

# 포트 51122에서 실행
api.run(host='0.0.0.0', port=51122, debug=True)
