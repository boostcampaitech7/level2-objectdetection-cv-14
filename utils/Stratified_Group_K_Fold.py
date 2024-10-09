import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter
import argparse

def get_distribution(y):
    y_distr = Counter(y)
    y_vals_sum = sum(y_distr.values())

    return [f'{y_distr[i]/y_vals_sum:.2%}' for i in range(np.max(y) +1)]

def main():
    parser = argparse.ArgumentParser()

    # k-fold 수 지정 인자
    parser.add_argument('-k', '--kfold', help="k-fold's k", type=int, default=5)

    # random seed 지정 인자
    parser.add_argument('-s', '--seed', help='random seed value', type=int, default=14)

    # 인자 받기
    args = parser.parse_args()

    # train.json 파일 절대 경로
    annotation = "/data/ephemeral/home/dataset/train.json"

    with open(annotation) as f: 
        data = json.load(f)
    
    # annotation 정보에서 image_id와 category_id를 가져와 튜플 구성 후 리스트로 반환
    var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]

    X = np.ones((len(data['annotations']), 1))
    y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var])

    cv = StratifiedGroupKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

    distrs = [get_distribution(y) + [len(np.unique(groups))]]

    index = ['training set']

    for fold_ind, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start = 1):
        train_y, val_y = y[train_idx], y[val_idx] 
        # group에서 중복을 제거하고 뒤에 있을 in 연산자에서 O(1)로 수행하기 위해 set으로 변환
        train_gr, val_gr = set(groups[train_idx]), set(groups[val_idx]) 

        assert len(train_gr & val_gr) == 0

        distrs.append(get_distribution(train_y) + [len(train_gr)])
        distrs.append(get_distribution(val_y) + [len(val_gr)])

        index.append(f'train - fold{fold_ind}')
        index.append(f'val - fold{fold_ind}')

        # train group에 해당하는 것만 가져와 list 구성
        train_imgs = [img for img in data['images'] if img['id'] in train_gr]
        train_data = {
            "images" : train_imgs, # train image 정보
            "categories" : data["categories"], #category 정보
            # train_idx에 해당하는 annotation들로 list 구성
            "annotations" : [data['annotations'][idx] for idx in train_idx] 
        }

        # val group에 해당하는 것만 가져와 list 구성
        val_imgs = [img for img in data['images'] if img['id'] in val_gr]
        val_data = {
            "images" : val_imgs, # val image 정보
            "categories" : data["categories"], #category 정보
            # val_idx에 해당하는 annotation들로 list 구성
            "annotations" : [data['annotations'][idx] for idx in val_idx]
        }

        # /data/ephemeral/home/dataset/sgkf_kfold_seed
        basic_path = os.path.join(os.path.split(annotation)[0], f"sgkf_{args.kfold}_{args.seed}")

        if not os.path.exists(basic_path):
            os.mkdir(basic_path)

        # /data/ephemeral/home/dataset/sgkf_kfold_seed/train_1fold.json
        train_path = os.path.join(basic_path, f"train_{fold_ind}fold.json")
        with open(train_path, "w") as f:
            json.dump(train_data, f, indent=4)
        
        # /data/ephemeral/home/dataset/sgkf_kfold_seed/val_1fold.json
        val_path = os.path.join(basic_path, f"val_{fold_ind}fold.json")
        with open(val_path, "w") as f:
            json.dump(val_data, f, indent=4)

    categories = [d['name'] for d in data['categories']]
    print(pd.DataFrame(distrs, index=index, columns = [categories[i] for i in range(np.max(y) + 1)] + ["image_cnt"]))
    
if __name__=="__main__":
    main()