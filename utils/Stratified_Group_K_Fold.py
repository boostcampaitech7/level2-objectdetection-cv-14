import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter
import argparse

def main():
    parser = argparse.ArgumentParser()

    # k-fold 수 지정 인자
    parser.add_argument('-k', '--kfold', help="k-fold's k", type=int, default=5)

    # random seed 지정 인자
    parser.add_argument('-s', '--seed', help='random seed value', type=int, default=14)

    # 인자 받기
    args = parser.parse_args()

if __name__=="__main__":
    main()