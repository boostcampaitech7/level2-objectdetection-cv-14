import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from ensemble_boxes import *
import matplotlib.pyplot as plt
import argparse


def main():
    csv_files = [os.path.join('./target', f)
                 for f in os.listdir('./target') if f.endswith('.csv')]
    combined_df = [pd.read_csv(f) for f in csv_files]

    image_ids = combined_df[0]['image_id'].tolist()
    assert len(image_ids) == 4871

    prediction_strings = []
    file_names = []

    img_width, img_height = 1024, 1024
    result = []
    for i, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []

        for df in combined_df:
            predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[
                0]
            predict_list = str(predict_string).split()

            if len(predict_list) <= 1:
                continue

            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []

            for box in predict_list[:, 2:6].tolist():
                box[0] = float(box[0]) / img_width
                box[1] = float(box[1]) / img_height
                box[2] = float(box[2]) / img_width
                box[3] = float(box[3]) / img_height
                box_list.append(box)

            boxes_list.append(box_list)
            # 1보다 큰 값이 있는지 확인
            large_value_exists = any(any(val > 1 for val in box)
                                     for box in box_list)
            if large_value_exists:
                print(f"경고: image_id {image_id}에서 1보다 큰 값이 발견되었습니다.")
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))

        # Ensemble 방법 선택
        iou_thr = 0.5
        if len(boxes_list):
            # 앙상블 방법 선택을 위한 인자 파싱
            parser = argparse.ArgumentParser()
            parser.add_argument('--ensemble_method', type=str, default='nms',
                                choices=['nms', 'soft_nms', 'wbf',
                                         'wbf2', 'nmw'],
                                help='앙상블 방법 선택')
            args = parser.parse_args()

            # 선택된 앙상블 방법에 따라 실행
            if args.ensemble_method == 'nms':
                boxes, scores, labels = nms(
                    boxes_list, scores_list, labels_list, iou_thr=iou_thr)
            elif args.ensemble_method == 'soft_nms':
                boxes, scores, labels = soft_nms(
                    boxes_list, scores_list, labels_list, iou_thr=iou_thr)
            elif args.ensemble_method == 'wbf':
                boxes, scores, labels = weighted_boxes_fusion(
                    boxes_list, scores_list, labels_list, iou_thr=iou_thr)
            elif args.ensemble_method == 'wbf_box_model_avg':
                boxes, scores, labels = weighted_boxes_fusion(
                    boxes_list, scores_list, labels_list, iou_thr=iou_thr, conf_type='box_and_model_avg')
            elif args.ensemble_method == 'nmw':
                boxes, scores, labels = non_maximum_weighted(
                    boxes_list, scores_list, labels_list, iou_thr=iou_thr)
            else:
                raise ValueError(f"알 수 없는 앙상블 방법: {args.ensemble_method}")

            result.append(
                {'boxes': boxes.tolist(), 'scores': scores.tolist(), 'labels': labels.tolist()})
            for box, score, label in zip(boxes, scores, labels):
                prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0] * img_width) + ' ' + str(
                    box[1] * img_height) + ' ' + str(box[2] * img_width) + ' ' + str(box[3] * img_height) + ' '

        prediction_strings.append(prediction_string)
        file_names.append(image_id)

    # 모든 이미지의 boxes 개수의 합 계산
    total_boxes = sum(len(res['boxes']) for res in result)
    print(f"모든 이미지의 boxes 개수 합계: {total_boxes}")

    # scores 통계값 계산 및 출력
    all_scores = [score for res in result for score in res['scores']]
    avg_score = sum(all_scores) / len(all_scores)
    min_score = min(all_scores)
    max_score = max(all_scores)
    median_score = sorted(all_scores)[len(all_scores) // 2]

    print(f"점수 통계:")
    print(f"  평균: {avg_score:.4f}")
    print(f"  최소값: {min_score:.4f}")
    print(f"  최대값: {max_score:.4f}")
    print(f"  중앙값: {median_score:.4f}")

    # 점수 분포를 그래프로 표현
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=50, edgecolor='black')
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')

    # 그래프 저장
    plt.savefig('score_distribution.png')
    print("점수 분포 그래프가 'score_distribution.png'로 저장되었습니다.")

    # 그래프 표시 (선택사항)
    # plt.show()

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names

    # csv file 경로 입력
    save_dir = './output'

    # 결과를 저장할 파일 이름 생성
    base_filename = 'ensemble.csv'
    counter = 1
    output_filename = base_filename

    while os.path.exists(os.path.join(save_dir, output_filename)):
        output_filename = f'ensemble_{counter}.csv'
        counter += 1

    # 결과 저장
    submission.to_csv(os.path.join(save_dir, output_filename), index=False)
    print(f'결과가 {output_filename}로 저장되었습니다.')


if __name__ == '__main__':
    main()
