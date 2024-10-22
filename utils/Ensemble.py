import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from ensemble_boxes import *


def main():
    csv_files = [os.path.join('./target', f)
                 for f in os.listdir('./target') if f.endswith('.csv')]
    combined_df = [pd.read_csv(f) for f in csv_files]

    image_ids = combined_df[0]['image_id'].tolist()
    assert len(image_ids) == 4871

    prediction_strings = []
    file_names = []

    img_width, img_height = 1024, 1024

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
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))

        # Ensemble 방법 선택
        iou_thr = 0.5
        if len(boxes_list):
            # boxes, scores, labels = nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr) # NMS
            # boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr) # Soft_NMS
            # boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=iou_thr) # WBF
            boxes, scores, labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list, iou_thr=iou_thr, conf_type='box_and_model_avg')  # WBF with box_and_model_avg
            # boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, iou_thr=iou_thr) # NMW

            for box, score, label in zip(boxes, scores, labels):
                prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0] * img_width) + ' ' + str(
                    box[1] * img_height) + ' ' + str(box[2] * img_width) + ' ' + str(box[3] * img_height) + ' '

        prediction_strings.append(prediction_string)
        file_names.append(image_id)

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
