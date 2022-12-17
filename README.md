# Anomaly Detection Simulator - Backend

## Framework

Flask

## 실행 방법

python 가상 환경 생성 후 아래의 명령어 실행

```
$ pip install -r requirements.txt
$ python app.py
```

기본 사용 포트: 51122

## 파일 구조

```
backend
├── cs_flow
│   ├── models
│   │   ├── lens
│   │   ├── flex
│   │   └── SMT
│   ├── evaluate_one.py
│   └── ...
├── static
│   ├── lens
│   │   ├── OK
│   │   └── NG
│   ├── flex
│   │   ├── OK
│   │   └── NG
│   ├── SMT
│   │   ├── OK
│   │   └── NG
│   ├── test_map
│   └── histogram
├── app.py
└── freia_funcs.py
```

- 각 모델은 cs_flow/models에 데이터셋 이름으로 저장해야 하며, 시뮬레이션 실행 시에 로드됩니다.
- evaluate 시에 생성되는 visualization, histogram 이미지는 static 디렉토리의 각 폴더에 저장됩니다.
- 데이터셋별 이미지는 static/데이터셋 디렉토리에 OK와 NG를 나누어 저장합니다.
- evaluate_one.py는 하나의 이미지만을 평가하기 위해 기존 cs-flow evaluate.py를 수정해 만든 파일입니다.
- freia_funcs.py 파일은 cs_flow 파일 중 하나로, 경로 문제로 인해 app.py와 같은 위치에 존재합니다.

## API 설명

### 1) /getAllData [GET]

arguments: dataset(데이터셋 종류)

static/dataset에 있는 모든 데이터의 리스트를 랜덤으로 순서를 섞어 배열로 반환합니다.

### 2) /predict [GET]

arguments: model(모델 종류, CS-Flow로 고정), dataset(데이터셋 종류), img_name(이미지 이름)

선택된 데이터셋의 모델을 cs_flow/models에서 불러오고, img_name에 해당하는 이미지에 대해 찾아 양/불량을 판단합니다.  
이미지 실제 label, 예측 결과, 원본 이미지, visualization 이미지를 반환합니다.

### 3) /getHistogram [GET]

arguments: dataset(데이터셋 종류)

예측이 진행된 모든 데이터의 score를 이용해 histogram을 생성하고, 이를 반환합니다.
