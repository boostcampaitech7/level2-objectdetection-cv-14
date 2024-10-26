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
