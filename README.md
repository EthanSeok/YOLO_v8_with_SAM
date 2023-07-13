## 딥러닝 - 포스트
* [딥러닝 포스트](https://ethanseok.github.io/tags/#%EB%AA%A8%EB%8D%B8%EB%A7%81)

<br>

참고 링크 (GitHub):  
[SAM-annotator](https://github.com/haochenheheda/segment-anything-annotator)  
[YOLO v8](https://github.com/ultralytics/ultralytics)


## YOLO v8이란?

Object Detection, 객체 인식은 이미지 또는 비디오에서 개체를 식별하고 찾는 것과 관련된 컴퓨터 비전 작업이다.
Object Detection 기술은 다음 두 가지 질문을 위해 존재한다.


* 이것은 무엇인가? 특정 이미지에서 대상을 식별하기 위함  
* 어디에 위치해있는가? 이미지 내에서 개체의 정확한 위치를 설정하기 위함


기존 객체 인식은 다양한 접근 방식으로 시도되면서 데이터 제한 및 모델링 문제를 해결하고자 하였다. 그리고 높은 검출 속도와 정확도를 보이는 **단일** 알고리즘 실행을 통해 객체를 감지할 수 있는 YOLO 알고리즘이 등장하게 되었다.

YOLO v8은 23년 7월 기준 가장 최근에 나온 YOLO 버전이며 2023 년 1월 Ultralytics 에서 개발되었다. 
YOLO 모델을 위한 완전히 새로운 리포지토리를 출시하여 개체 감지, 인스턴스 세분화 및 이미지 분류 모델을 train하기 위한 통합 프레임워크로 구축되었다.


자세한 내용은 아래 링크에 잘 설명되어 있으니 참고하면 좋을 것.  
[참고링크1](https://www.thedatahunt.com/trend-insight/guide-for-yolo-object-detection)  
[참고링크2](https://velog.io/@qtly_u/n4ptcz54#span-style--color-cornflowerblue-yolo)

<br>

## SAM

SAM (Sagment Anything Model)이란 무엇인지는 ([참고 자료](https://ethanseok.github.io/2023-04-30/sam-post))를 읽어보면 알 수 있다.

<br>

이름| 원본 데이터 (개) | Train (개) | Validation (개) |
---|------------|-----------|----------------|
진딧물| 742        | 575       | 144            |
가루이-약충| 615        | 490       | 123            |
가루이-성충| 708        | 566       | 142            |
전체| 2,065      | 1,631     | 409       |

<br>

**증강후** 최종적으로 YOLO 학습에 사용한 데이터 정보는 다음과 같다.

이름| 증강 후 데이터 (개) | 증강 후 Train (개) | 증강 후 Validation (개) |
---|--------------|----------------|---------------------|
전체| 5,749        | 4,566          | 1,183               |


<br>

학습을 위한 데이터는 총 데이터의 8:2 비율로 학습 및 검증에 사용할 것이다. 다음은 학습에 사용한 데이터 정보이다.

진딧물 (aphid)| 가루이-약충 | 가루이-성충 |
---|--------|-----------|
![20220811-134103](https://github.com/EthanSeok/yolov5_detection/assets/93086581/0d406b6e-b950-4792-9dce-b0282c5b155e)|![20220830-151101](https://github.com/EthanSeok/yolov5_detection/assets/93086581/048f453f-427e-448d-b524-c1edeef8c603)| ![20220824-171308](https://github.com/EthanSeok/yolov5_detection/assets/93086581/21d0c031-395f-4272-b1c7-582935178767)|   

<br>

## YOLO 학습 결과

**Confusion Matrix**

![confusion matrix](https://github.com/EthanSeok/yolov5_detection/assets/93086581/55f66baa-134e-49f8-8b55-d91b4a7f0c86)

<br>

**Train & Validation Loss and Precision**

![graph](https://github.com/EthanSeok/yolov5_detection/assets/93086581/2ecf4693-e7a1-4b33-87d0-ba8d2bf7ec55)

<br>

**F1-score**

![f1_score](https://github.com/EthanSeok/yolov5_detection/assets/93086581/1f58b536-5d8e-4e81-b181-f36d03bdf9c6)

<br>

## SAM 결합 결과

진딧물 (aphid)| 가루이-약충                                                                                                         | 가루이-성충                                                                                                         |
---------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
![image](https://github.com/EthanSeok/YOLO_v8_with_SAM/assets/93086581/142e9c68-4564-4675-a894-cac0d38009f4) | ![image](https://github.com/EthanSeok/YOLO_v8_with_SAM/assets/93086581/fe4c596b-c661-4947-8394-5d7ace8b0250) |![image](https://github.com/EthanSeok/YOLO_v8_with_SAM/assets/93086581/10314207-a2e8-4843-b763-d2d0c2715976)  |
