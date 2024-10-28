# 📋 Project Overview


![project_image](https://github.com/user-attachments/assets/a15ac710-0ed3-496b-9e86-00213727cde5)

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다.

- 입력
  - **쓰레기 객체가 담긴 이미지, bbox 정보(좌표, 카테고리)**
  - bbox annotation은 **COCO format**
- 출력
  - **bbox 좌표, 카테고리, score** 값을 리턴.
  - submission 양식에 맞게 csv 파일을 만들어 제출
  - COCO format이 아닌 **Pascal VOC format**

<br/>

# 🗃️ Dataset

- 전체 이미지
  - **9754 images**
  - train
    - **4883 images**
  - test
    - **4871 images**
- 클래스 수
  - **10 class**
  - General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기
  - **1024 x 1024**

<br/>

# 😄 Team Member

<table align="center">
    <tr align="center">
        <td><img src="https://github.com/user-attachments/assets/3560856a-8cac-4079-8494-4f1bf13d0eb5" width="140"></td>
        <td><img src="https://github.com/user-attachments/assets/3560856a-8cac-4079-8494-4f1bf13d0eb5" width="140"></td>
        <td><img src="https://github.com/user-attachments/assets/3560856a-8cac-4079-8494-4f1bf13d0eb5" width="140"></td>
        <td><img src="https://github.com/user-attachments/assets/3560856a-8cac-4079-8494-4f1bf13d0eb5" width="140"></td>
        <td><img src="https://github.com/user-attachments/assets/fdce3bf1-4dd2-44c9-b4db-1599c4d3826d" width="140" height="140"></td>
        <td><img src="https://github.com/user-attachments/assets/3560856a-8cac-4079-8494-4f1bf13d0eb5" width="140"></td>
    </tr>
    <tr align="center">
        <td><a href="https://github.com/kimgeonsu" target="_blank">김건수</a></td>
        <td><a href="https://github.com/202250274" target="_blank">박진영</a></td>
        <td><a href="https://github.com/oweixx" target="_blank">방민혁</a></td>
        <td><a href="https://github.com/lkl4502" target="_blank">오홍석</a></td>
        <td><a href="https://github.com/Soy17" target="_blank">이소영</a></td>
        <td><a href="https://github.com/yejin-s9" target="_blank">이예진</a></td>
    </tr>
    <tr align="center">
        <td>T7103</td>
        <td>T7156</td>
        <td>T7158</td>
        <td>T7208</td>
        <td>T7222</td>
        <td>T7225</td>
    </tr>
</table>

<br/>

# 🗂️ Role

| Member |                                            Role                                            |
| :----: | :----------------------------------------------------------------------------------------: |
| 김건수 |                            PM 역할 수행, YOLO Develop, Ensemble                            |
| 박진영 |                         ConvNeXT Develop, Ensemble, Util 기능 구현                         |
| 방민혁 |                       EDA, Data Relabel, ATSS Swin Develop, Ensemble                       |
| 오홍석 | PM 역할 수행, DETR 기반 Model Develop, <br> Util 기능 구현, Project 구조 및 서버 환경 구성 |
| 이소영 |                        VFNet, RTMDet Model Develop, Util 기능 구현                         |
| 이예진 |                          EDA, Data Relabel, Pseudo Labeling, TTA                           |

<br/>

# 🧳 Project Progress Summary

아래의 항목들로 프로젝트를 진행한 과정을 설명한다.

<br>

## Project Structure

```
📦 level2-objectdetection-cv-14
┣ 📂 EDA_data
┃ ┣ 📜 eda_(2).ipynb
┃ ┣ 📜 eda_dataset.ipynb
┃ ┣ 📜 eda_traindata.ipynb
┣ 📂 mmdetection
┃ ┣ 📂 configs
┃ ┣ 📂 custom_configs
┃ ┣ 📂 mmdet
┃ ┗ 📜 train.py
┃ ┗ 📜 inference.py
┣ 📂 mmdetectionV3
┃ ┣ 📂 configs
┃ ┣ 📂 custom_configs
┃ ┣ 📂 mmdet
┃ ┗ 📜 train.py
┃ ┗ 📜 inference.py
┣ 📂 yolo
┣ 📂 utils
┃ ┣ 📜 csv_pseudo.py
┃ ┣ 📜 Ensemble.py
┃ ┣ 📜 Gsheet.py
┃ ┣ 📜 modify_test.py
┃ ┣ 📜 pseudo_data_split.py
┃ ┣ 📜 pseudo_ensemble_labeling.py
┃ ┣ 📜 pseudo_labeling.py
┃ ┣ 📜 pseudo_labes_count.py
┃ ┣ 📜 split_val_train_log.py
┃ ┣ 📜 Stratified_Group_K_Fold.py
┣ 📃 requirements.txt
┗ 📃 README.md
```

<br>

## 🕵🏻 EDA

Class Imbalance, Object Size 등 여러 항목에 대해서 진행하였다.  
아래는 그 중 하나에 대한 예시이다.  
[Wrap-UP Report 참고](#-object-detection-wrap-up-report)

> ### BBox Area Distribution

![image](https://github.com/user-attachments/assets/17fbdea6-f2e2-4ee6-acca-45ea9a0cb2d3)

- Clothing은 상대적으로 큰 박스 크기를 가지고 있으며 Battery와 같은 물체는 작고 일정한 크기로 나타나는 경향을 확인
- Glass, Plastic, Paper Pack, Plastic bag 등의 경우, 박스 크기가 매우 다양한 분포를 보임

<br>

## 🧪 Experiments

진행한 실험으로는 Data Relabeling, Pseudo Labeling 등이 있다.  
아래는 그 중 하나에 대한 예시이다.  
[Wrap-UP Report 참고](#-object-detection-wrap-up-report)

> ### Data Relabeling

<img src="https://github.com/user-attachments/assets/44948f78-f7f2-4215-af5f-b3d47e67d9e4" height="300">  
<img src="https://github.com/user-attachments/assets/5e0ed86d-04d7-4631-854f-e52b8b78d73e" height="300">

- 잘못된 라벨링에 대해서 수정하는 작업을 거침
- Object 마다 label의 통일성 유지 ex) 전단지나 명함 같은 경우 General trash로 통일

<br>

## 📚 Model Selection and Develop

사용한 모델에는 ATSS Swin, ConvNeXT, DINO 등등 여러가지가 있다.  
아래는 그 중 하나에 대한 예시이다.  
[Wrap-UP Report 참고](#-object-detection-wrap-up-report)

> ### ATSS Swin

| Version |                  Description                   | Public mAP 50 |
| :-----: | :--------------------------------------------: | :-----------: |
|    1    |           ATSS Swin Base model 적용            |    0.5587     |
|    2    |  pretrained model 교체 (swin win12-384 model)  |    0.5397     |
|    3    |          Cascade Swin Base model 적용          |    0.5482     |
|    4    |       load_from(사전 학습된 가중치) 적용       |    0.6297     |
|    5    |               Anchor ratios 수정               |    0.6073     |
|    6    | train_pipeline의 Resize를 multiscale_v1로 수정 |    0.6536     |
|    7    |               Hard Augmentation                |       0       |
|    8    |  train_pipeline의 Resize를 (1024,1024)로 수정  |    0.6558     |
|    9    | train_pipeline의 Resize를 multiscale_v2로 수정 |    0.6800     |

<br>

## 🖇️ Ensemble

- 모델간 앙상블에서 2가지 전략을 사용하였다.
- Stratified Group K Fold Cross Validation
  - 각기 다른 Fold에 학습한 같은 구조의 모델간 앙상블 (NMS, WBF)
- 다른 모델간 앙상블
  - Confusion Matrix와 같은 평가 지표를 활용하여 모델간 특성을 파악
  - 파악한 모델간 특성을 바탕으로 최적의 모델 조합 선택

> ### Stratified Group K Fold Cross Validation Ensemble

|   Model   | Fold Avg Score |  WBF   |  NMS   |
| :-------: | :------------: | :----: | :----: |
| ConvNeXT  |     0.6929     | 0.7091 | 0.7063 |
|   DINO    |     0.6969     | 0.5328 | 0.7106 |
| ATSS Swin |     0.6791     | 0.6929 | 0.6970 |
|   YOLO    |     0.4360     | 0.5539 | 0.5272 |
|  CO-DINO  |     0.6955     | 0.6212 | 0.7111 |

<br>

> ### Final Model Ensemble Strategy

![image](https://github.com/user-attachments/assets/6e1b4d3f-deed-4f10-8c53-c48d61d69164)

<br>

## Utils

프로젝트를 진행하면서 편의성을 위한 기능 또는 실험을 위한 추가 기능들을 구현하였다.

- Stratified Group K Fold Cross Validation
- Google Sheet을 이용한 실험 인자 기록 자동화
- Pseudo Labeling 관련 기능
- train / inference log 분할 기능 등등

아래는 그 중 하나에 대한 예시이다.  
[Notion 참고](#-object-detection-notion)

> ### Stratified Group K Fold Cross Validation
>
> - 학습한 모델의 성능 평가를 위해서 Validation Set을 분리해낸다.
> - 기존 데이터 셋의 클래스 분포를 유지한다. (아래 표 참고)
> - 같은 이미지에서 나온 annotation이 Train 또는 Validation에만 포함되도록 구분한다.

<table align="center">
  <thead align="center">
    <tr>
      <th></th>
      <th>General trash</th>
      <th>Paper</th>
      <th>Paper pack</th>
      <th>Metal</th>
      <th>Glass</th>
      <th>Plastic</th>
      <th>Styrofoam</th>
      <th>Plastic bag</th>
      <th>Battery</th>
      <th>Clothing</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <th nowrap>training set</th>
      <td>17.14%</td>
      <td>27.45%</td>
      <td>3.88%</td>
      <td>4.04%</td>
      <td>4.24%</td>
      <td>12.72%</td>
      <td>5.46%</td>
      <td>22.37%</td>
      <td>0.69%</td>
      <td>2.02%</td>
    </tr>
    <tr>
      <th nowrap>train - fold1</th>
      <td>17.11%</td>
      <td>26.73%</td>
      <td>3.92%</td>
      <td>4.07%</td>
      <td>4.14%</td>
      <td>13.01%</td>
      <td>5.46%</td>
      <td>22.88%</td>
      <td>0.66%</td>
      <td>2.02%</td>
    </tr>
    <tr>
      <th>val - fold1</th>
      <td>17.23%</td>
      <td>29.88%</td>
      <td>3.72%</td>
      <td>3.96%</td>
      <td>4.59%</td>
      <td>11.70%</td>
      <td>5.45%</td>
      <td>20.66%</td>
      <td>0.76%</td>
      <td>2.04%</td>
    </tr>
    <tr>
      <th nowrap>train - fold2</th>
      <td>17.17%</td>
      <td>27.75%</td>
      <td>3.92%</td>
      <td>4.16%</td>
      <td>4.27%</td>
      <td>12.52%</td>
      <td>5.47%</td>
      <td>22.21%</td>
      <td>0.66%</td>
      <td>1.88%</td>
    </tr>
    <tr>
      <th>val - fold2</th>
      <td>17.01%</td>
      <td>26.23%</td>
      <td>3.72%</td>
      <td>3.58%</td>
      <td>4.13%</td>
      <td>13.53%</td>
      <td>5.40%</td>
      <td>23.02%</td>
      <td>0.81%</td>
      <td>2.58%</td>
    </tr>
    <tr>
      <th nowrap>train - fold3</th>
      <td>17.05%</td>
      <td>27.66%</td>
      <td>3.95%</td>
      <td>3.81%</td>
      <td>4.37%</td>
      <td>12.35%</td>
      <td>5.67%</td>
      <td>22.51%</td>
      <td>0.66%</td>
      <td>1.99%</td>
    </tr>
    <tr>
      <th>val - fold3</th>
      <td>17.53%</td>
      <td>26.55%</td>
      <td>3.56%</td>
      <td>5.04%</td>
      <td>3.71%</td>
      <td>14.29%</td>
      <td>4.56%</td>
      <td>21.79%</td>
      <td>0.82%</td>
      <td>2.16%</td>
    </tr>
    <tr>
      <th nowrap>train - fold4</th>
      <td>17.18%</td>
      <td>27.19%</td>
      <td>3.85%</td>
      <td>4.00%</td>
      <td>4.28%</td>
      <td>12.66%</td>
      <td>5.66%</td>
      <td>22.37%</td>
      <td>0.70%</td>
      <td>2.11%</td>
    </tr>
    <tr>
      <th>val - fold4</th>
      <td>16.96%</td>
      <td>28.54%</td>
      <td>4.00%</td>
      <td>4.23%</td>
      <td>4.07%</td>
      <td>12.94%</td>
      <td>4.59%</td>
      <td>22.40%</td>
      <td>0.64%</td>
      <td>1.64%</td>
    </tr>
    <tr>
      <th nowrap>train - fold5</th>
      <td>17.18%</td>
      <td>27.88%</td>
      <td>3.75%</td>
      <td>4.19%</td>
      <td>4.15%</td>
      <td>13.05%</td>
      <td>5.02%</td>
      <td>21.92%</td>
      <td>0.76%</td>
      <td>2.11%</td>
    </tr>
    <tr>
      <th>val - fold5</th>
      <td>16.95%</td>
      <td>25.66%</td>
      <td>4.41%</td>
      <td>3.46%</td>
      <td>4.63%</td>
      <td>11.33%</td>
      <td>7.23%</td>
      <td>24.25%</td>
      <td>0.40%</td>
      <td>1.68%</td>
    </tr>
  </tbody>
</table>

<br>

## 🏆 Project Result

**_<p align=center>Public Leader Board</p>_**
<img src="https://github.com/user-attachments/assets/659d0f34-f546-4a6f-9939-d254bcb98a15" alt="Public Leader Board" >

<br>

**_<p align=center>Private Leader Board</p>_**
<img src="https://github.com/user-attachments/assets/d870f505-96db-4dee-bd16-ab9e03f7ba66" alt="Private Leader Board" >

<br>

## 🔗 Reference

### [📎 Object Detection Wrap-UP Report](https://drive.google.com/file/d/1MvqASckPwXHHoGqNLIGiQHgjQBBYPDgK/view?usp=sharing)

### [📎 Object Detection Notion](https://violet-join-36b.notion.site/Recycle-Object-Detection-f114581f9bae41faba6cd302474f02d5?pvs=4)

<br>

## Commit Convention

1. `Feature` : **새로운 기능 추가**
2. `Fix` : **버그 수정**
3. `Docs` : **문서 수정**
4. `Style` : **코드 포맷팅 → Code Convention**
5. `Refactor` : **코드 리팩토링**
6. `Test` : **테스트 코드**
7. `Comment` : **주석 추가 및 수정**

커밋할 때 헤더에 위 내용을 작성하고 전반적인 내용을 간단하게 작성합니다.

### 예시

- `git commit -m "[#issue] Feature : message content"`

커밋할 때 상세 내용을 작성해야 한다면 아래와 같이 진행합니다.

### 예시

> `git commit`  
> 어떠한 에디터로 진입하게 된 후 아래와 같이 작성합니다.  
> `[header]: 전반적인 내용`  
> . **(한 줄 비워야 함)**  
> 상세 내용

<br/>

## Branch Naming Convention

브랜치를 새롭게 만들 때, 브랜치 이름은 항상 위 `Commit Convention`의 Header와 함께 작성되어야 합니다.

### 예시

- `Feature/~~~`
- `Refactor/~~~`
