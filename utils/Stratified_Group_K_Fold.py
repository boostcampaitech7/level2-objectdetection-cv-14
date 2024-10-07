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

    annotation = "/data/ephemeral/home/dataset/train.json"

    with open(annotation) as f: 
        data = json.load(f)

    var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]

    X = np.ones((len(data['annotations']), 1))
    y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var])

    cv = StratifiedGroupKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

    distrs = [get_distribution(y) + [len(np.unique(groups))]]

    index = ['training set']

    for fold_ind, (train_idx, val_idx) in enumerate(cv.split(X,y, groups)):
        train_y, val_y = y[train_idx], y[val_idx]
        train_gr, val_gr = groups[train_idx], groups[val_idx]

        assert len(set(train_gr) & set(val_gr)) == 0
        distrs.append(get_distribution(train_y) + [len(np.unique(train_gr))])

        distrs.append(get_distribution(val_y) + [len(np.unique(val_gr))])
        index.append(f'train - fold{fold_ind}')
        index.append(f'val - fold{fold_ind}')

    categories = [d['name'] for d in data['categories']]
    print(pd.DataFrame(distrs, index=index, columns = [categories[i] for i in range(np.max(y) + 1)] + ["image_cnt"]))

if __name__=="__main__":
    main()