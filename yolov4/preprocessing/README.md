# 0.concat_all_is_eval
 

NIA Json 들을 훈련을 위해 하나로 합치는 과정

 

NIA dataset이 수퍼카테고리 별로 json이 구분되어 있으므로 concat이 필요

수퍼카테고리 별로 index이 되어 있기 때문에 데이터를 합쳐도 고유한 index 필요

(josn_idx) = super category, 

train_cls_idx = 고유한 클래스 index, 

train_img_idx = 고유한 이미지 index

train과 eval을 위해 각 수퍼카테고리별 dataset의 20%를 eval로 라벨링 필요

현재 super class의 20%만 eval로 분류하여 eval 데이터에 하위 클래스가 누락되는 경우가 발생

ex) money category의 100원 50원은 eval에 있는데 500원은 없음 (모두 train으로 감)

# 1.nia2yolo
 

NIA 데이터셋을 yolo 모델에 학습이 가능하게 필요한 컬럼들을 추가하는 작업

yolo bbox는 [ center_x, center_y, width. height] 형태로 구성

 

bbox >>> x,y,w,h 나누기

image height, width 추가

위에 작업한 내용들을 이용하여 nia2yolo 함수를 이용하여 yolo bbox 추가

 

# 2.labels
 

yolo 학습할 이미지의 라벨을 txt 파일로 읽으므로 txt 파일 생성 필요

 

images json으로 부터 이미지의 이름을 가져와 annotation json에서 해당 이름을 갖는 라벨링을 모두 찾은 후 class_label과 yolo bbox를 가져와 저장

즉, 이미지 이름명.txt  파일에 이미지 안에 들어 있는 모든 라벨 정보가 기록 됨 

eval 폴더와 train 폴더 따로 만들어 각 사용될 이미지.txt 저장

 

 

# 3.file_list
 

train에 사용될 이미지 이름들이 들어있는 txt 파일과

eval에 사용될 이미지 이름들이 들어있는 txt 파일 생성

 

경로를 labels에서 저장한 dir과 동일한 형태로 맞춰야 함
