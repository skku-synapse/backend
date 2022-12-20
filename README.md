# Anomaly Detection Simulator - Backend

> 렌즈 데이터셋, Flex 데이터셋, SMT 데이터셋에 대한 CS-Flow 모델의 예측 결과를 실시간으로 확인할 수 있는 웹 어플리케이션입니다.  
> 데이터 예측 결과, 실제 양/불량 여부, visualization 결과를 확인할 수 있으며 미검율, 과검율, score histogram 등의 분석 결과를 제공합니다.

## Framework

`Flask`

## Getting Started

### Clone Repository

```
$ git clone https://github.com/skku-synapse/backend.git
$ cd backend
```

### How to Run

**Installation:**

```
$ python -m pip install -U pip
$ pip install -r requirements.txt
```

**To run Flask:**

```
$ sudo python app.py
```

## 파일 구조

```
.
├── cs_flow/
│   ├── models/
│   │   ├── lens
│   │   ├── flex
│   │   └── SMT
│   ├── evaluate_one.py
│   └── ...
├── static/
│   ├── lens/
│   │   ├── OK/
│   │   └── NG/
│   ├── flex/
│   │   ├── OK/
│   │   └── NG/
│   ├── SMT/
│   │   ├── OK/
│   │   └── NG/
│   ├── test_map/
│   └── histogram/
├── app.py
└── freia_funcs.py
```

- 모든 api는 app.py에 정의되어 있음.
- 각 모델은 cs_flow/models에 데이터셋 이름으로 저장해야 하며, 시뮬레이션 실행 시에 로드됨.
- evaluate 시에 생성되는 visualization, histogram 이미지는 static 디렉토리의 각 폴더에 저장됨.
- 데이터셋별 이미지는 static/데이터셋 디렉토리에 OK와 NG를 나누어 저장함.
- evaluate_one.py는 하나의 이미지만을 평가하기 위해 기존 cs-flow evaluate.py를 수정해 만든 파일임.
- freia_funcs.py 파일은 cs_flow 파일 중 하나로, 경로 문제로 인해 app.py와 같은 위치에 존재함.

## API 설명

### 1) /getAllData [GET]

static/dataset에 있는 모든 데이터의 리스트를 랜덤으로 순서를 섞어 배열로 반환  
예측 결과를 저장할 파일 생성, 이미 존재할 경우 초기화됨 (파일 이름: `[dataset]_result.csv`)

arguments: dataset(데이터셋 종류)

**Request 예시**

```
{
    method: "get",
    url: "http://[서버주소]:[포트번호]/getAllData",
    params: {
        dataset: "lens",
    },
    headers: {
        "Content-Type": "multipart/form-data",
    }
}
```

**Response 예시**

```
{
    list: [OK/image1.jpg, NG/image3.jpg, OK/image2.jpg]
}
```

### 2) /predict [GET]

선택된 데이터셋의 모델을 cs_flow/models에서 불러오고, img_name에 해당하는 이미지에 대해 양/불량을 판단  
이미지 실제 label, 예측 결과, 원본 이미지, visualization 이미지를 반환  
예측 결과를 (dataset)\_result.csv 파일에 추가

**파일 예시**

```
file name, label, prediction, anomaly score
image1.jpg, NG, NG, 3.99
image2.jpg, NG, OK, 1.00
...
```

arguments: model(모델 종류, CS-Flow로 고정), dataset(데이터셋 종류), img_name(이미지 이름)

**Request 예시**

```
{
    method: "get",
    url: "http://[서버주소]:[포트번호]/predict",
    params: {
        model: "CS-Flow",
        dataset: "lens",
        img_name: "OK/image1.jpg",
    },
    headers: {
        "Content-Type": "multipart/form-data",
    },
}
```

**Response 예시**

```
{
    label: "OK",
    isAnomaly: false,
    image: 원본 이미지,
    overlay: visualization 이미지
}
```

### 3) /getHistogram [GET]

예측이 진행된 모든 데이터의 score를 이용해 histogram을 생성하고, 이를 반환

arguments: dataset(데이터셋 종류)

**Request 예시**

```
{
    method: "get",
    url: "http://[서버주소]:[포트번호]/getHistogram",
    params: {
        dataset: dataset,
    },
    headers: {
        "Content-Type": "multipart/form-data",
    },
}
```

**Response 예시**

```
{
    status: true,
    image: 히스토그램 이미지
}
```
