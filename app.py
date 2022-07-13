import base64
import sys, os
import json
from flask import Flask, make_response, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import torch
#from backend.cs_flow.evaluate_one import evaluate_function

from cflow import *
#from cs_flow import *
from cs_flow.evaluate_one import evaluate_function
# from simulator.backend.fastflow.synapse_main import evaluate_image, parse_args
# from cs_flow_X import *
from patchcore_inspection.src import *

from setproctitle import *

os.environ["CUDA_VISIBLE_DEVICES"]="3"
setproctitle('synapse-backend')

api = Flask(__name__)
CORS(api, support_credentials=True)

@api.route('/')
def my_profile():
    response_body = {
        "name": "Jinsuk",
        "about" :"Hello! I'm a full stack developer that loves python and javascript"
    }

    return response_body


@api.route('/getDataList', methods=['GET'])
def getDataList():
    dataset = request.args.get('dataset')
    root_dir = '/home/synapse/simulator/backend/static/' + dataset # 디렉토리
 
    ok_list = []
    ng_list = []
    possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.bmp', '.png'] # 이미지 확장자들
 
    for (root, dirs, files) in os.walk(root_dir):
        if len(files) > 0:
            for file_name in files:
                if os.path.splitext(file_name)[1] in possible_img_extension:
                    #img_path = root + '/' + file_name
                    img_path = file_name
                    # 경로에서 \를 모두 /로 바꿔줘야함
                    img_path = img_path.replace('\\', '/') # \는 \\로 나타내야함
                    if (root[-2:] == "OK"):
                        ok_list.append("OK/"+img_path)
                    else:
                        ng_list.append("NG/"+img_path)
                    

    return {"ok": ok_list, "ng": ng_list}

@api.route('/getImage', methods=['GET'])
def getImage():
    dataset = request.args.get('dataset')
    img_name = request.args.get('img_name')
    user_img_path = './static/'+ dataset + '/' + img_name
    with open(user_img_path, "rb") as f:
        image_binary = f.read()

        response = make_response(base64.b64encode(image_binary))
        response.headers.set('Content-Type', 'image/jpg')
        response.headers.set('Content-Disposition', 'attachment', filename='image.jpg')
        image = base64.b64encode(image_binary).decode("utf-8")
        return jsonify({'status': True, 'image': image})

                    
@api.route('/predict', methods=['GET'])
def predict():
    print('start!')
    model_name = request.args.get('model')
    dataset = request.args.get('dataset') # module / lens / flex
    img_name = request.args.get('img_name')
    print(model_name)
    print(dataset)
    user_img_path = '/home/synapse/simulator/backend/static/'+ dataset + '/' + img_name

    if model_name == "CS-Flow":
        if dataset == "module":
            response = evaluate_function(user_img_path)
        elif dataset == "lens":
            response = evaluate_function(user_img_path, "camera_lens")
        elif dataset == "flex":
            response = evaluate_function(user_img_path, "board")
        return json.dumps(response)
    elif model_name == "CFLOW-AD":
        if dataset == "module":
            c = get_args('synapse', '/home/synapse/simulator/backend/cflow/weights_TestCropped/synapse_mobilenet_v3_large_freia-cflow_pl3_cb8_inp256_run0_SMT_OneClassLearning_DATASET_2022-06-16-14:37:24.pt', "SMT_OneClassLearning_DATASET_220620")
        elif dataset == "lens":
            c = get_args('camera', '/home/synapse/cflow-ad/weights/camera_mobilenet_v3_large_freia-cflow_pl3_cb8_inp512_run0_TM_BL_T1_CT_OneClassLearning_Dataset_220620_2022-06-24-09:38:40.pt' ,'TM_BL_T1_CT_OneClassLearning_Dataset_220620')
        elif dataset == "flex":
            c = get_args('new_data', '/home/synapse/cflow-ad/weights/new_data_mobilenet_v3_large_freia-cflow_pl3_cb8_inp512_run0_TM_FP_T3_DT_OneClassLearning_Dataset_220620_2022-06-28-20:48:50.pt' ,'TM_FP_T3_DT_OneClassLearning_Dataset_220620')
        response = test_one_image(c, dataset, img_name)
        return json.dumps(response)
    #elif model_name == "Fastflow":
    #    if dataset == "module":
    #        args = parse_args(dataset, "SMT_OneClassLearning_DATASET_220620")
    #    elif dataset == "lens":
    #        args = parse_args(dataset, 'TM_BL_T1_CT_OneClassLearning_Dataset_220620', '_fastflow_experiment_checkpoints/TM_BL/799.pt')
    #    elif dataset == "flex":
    #        args = parse_args(dataset, 'TM_FP_T3_DT_OneClassLearning_Dataset_220620', '_fastflow_experiment_checkpoints/TM_FP/449.pt')
    #    response = evaluate_image(args)
    #    return json.dumps(response)
    elif model_name == "PatchCore":
        device = torch.device('cuda:0')
        response = pipeline.main(device, user_img_path, dataset)
        print(response)
        return json.dumps(response)
    else:
        res = {
        'image': user_img_path,
        'isAnomaly': False,
        }
        return json.dumps(res)


@api.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        print('start!')
        user_img = request.files.get('file')
        fileName = request.form.get('fileName')
        model_name = request.args.get('model')
        dataset = request.args.get('dataset') # module / lens / flex
        print(model_name)
        print(dataset)
        user_img.save('./static/images/'+str(user_img.filename))
        user_img_path = './static/images/'+str(user_img.filename)

        if model_name == "CFLOW-AD":
            if dataset == "module":
                c = get_args('synapse', '/home/synapse/simulator/backend/cflow/weights_TestCropped/synapse_mobilenet_v3_large_freia-cflow_pl3_cb8_inp256_run0_SMT_OneClassLearning_DATASET_2022-06-16-14:37:24.pt', "SMT_OneClassLearning_DATASET_220620")
            elif dataset == "lens":
                c = get_args('camera', '/home/synapse/cflow-ad/weights/camera_mobilenet_v3_large_freia-cflow_pl3_cb8_inp512_run0_TM_BL_T1_CT_OneClassLearning_Dataset_220620_2022-06-24-09:38:40.pt' ,'TM_BL_T1_CT_OneClassLearning_Dataset_220620')
            elif dataset == "flex":
                c = get_args('new_data', '/home/synapse/cflow-ad/weights/new_data_mobilenet_v3_large_freia-cflow_pl3_cb8_inp512_run0_TM_FP_T3_DT_OneClassLearning_Dataset_220620_2022-06-28-20:48:50.pt' ,'TM_FP_T3_DT_OneClassLearning_Dataset_220620')
            response = test_one_image(c,dataset, fileName)
            return json.dumps(response)
        # elif model_name == 'PatchCore':
        #   response = pipeline.main()
        #   print(response)
        #   return json.dumps(response)
        # elif model_name == "CS-Flow":
        #     if dataset == "module":
        #         response = evaluate_function(user_img_path)
        #     elif dataset == "lens":
        #         response = evaluate_function(user_img_path, "camera_lens")
        #     elif dataset == "flex":
        #         response = evaluate_function(user_img_path, "board")
        #     return json.dumps(response)
        # elif model_name == "Fastflow":
        #     if dataset == "module":
        #         args = parse_args(dataset, user_img_path)
        #     elif dataset == "lens":
        #         args = parse_args(dataset, user_img_path, '_fastflow_experiment_checkpoints/TM_BL/799.pt')
        #     elif dataset == "flex":
        #         args = parse_args(dataset, user_img_path, '_fastflow_experiment_checkpoints/TM_FP/449.pt')
        #     response = evaluate_image(args)
        #     return json.dumps(response)
        else:
            res = {
            'image': user_img_path,
            'isAnomaly': False,
            }
            return json.dumps(res)

api.run(host='0.0.0.0', port=51122, debug=True)

 