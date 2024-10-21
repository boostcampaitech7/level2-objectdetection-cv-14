import os.path as osp
from typing import Sequence

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist

from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample
import pandas as pd

@HOOKS.register_module()
class SubmissionHook(Hook):
    """
    Args:
        prediction_strings (list): [labels + ' ' + scores + ' ' + x_min + ' ' + y_min + ' ' + x_max + ' ' + y_max]를 추가한 list
        file_names (list): img_path를 추가한 list
        test_out_dir (str) : 저장할 경로
    """

    def __init__(self, test_out_dir=None, output_name="submission.csv",thr=0.0):
        self.prediction_strings = []
        self.file_names = []
        self.test_out_dir = test_out_dir
        self.thr = thr
        self.output_file = output_name

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:
        """
        매 iter이 끝날때마다 수행
        Args:
            runner (:obj:`Runner`): 테스트를 수행하는 runner
            batch_idx (int): 최근 배치 index
            data_batch (dict): dataloader에서 온 data
            outputs (Sequence[:obj:`DetDataSample`]): 결과
        """
        assert len(outputs) == 1, \
            'only batch_size=1 is supported while testing.'

        for output in outputs:
            prediction_string = ''
            for label, score, bbox in zip(output.pred_instances.labels, output.pred_instances.scores, output.pred_instances.bboxes):
                bbox = bbox.cpu().numpy()
                # 이미 xyxy로 되어있음
                if score > self.thr:
                    prediction_string += str(int(label.cpu())) + ' ' + str(float(score.cpu())) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + ' '
            self.prediction_strings.append(prediction_string)
            self.file_names.append(output.img_path[-13:])

    def after_test(self, runner: Runner):        
        mkdir_or_exist(self.test_out_dir)
        submission = pd.DataFrame()
        submission['PredictionString'] = self.prediction_strings
        submission['image_id'] = self.file_names
        submission.to_csv(osp.join(self.test_out_dir, self.output_file), index=None)
        print('submission saved to {}'.format(osp.join(self.test_out_dir, self.output_file)))