import gspread
from gspread.exceptions import WorksheetNotFound
import json
from dotenv import dotenv_values
import os

# json 파일이 위치한 경로를 값으로 줘야 합니다.
def Gsheet_param(cfg):
    env_path = "/data/ephemeral/home/dataset/.env"
    env = dotenv_values(env_path)

    # train log 파일 경로 설정
    log_file_path = "JYP/level2-objectdetection-cv-14/mmdetection/work_dirs/swin_v4.log.json"

    # 서비스 연결
    gc = gspread.service_account(env['JSON_PATH'])

    # url에 따른 spread sheet 열기
    doc = gc.open_by_url(env['URL'])

    # model type
    worksheet_name = cfg['model']['type']
    
    # Samples per GPU
    samples_per_gpu = cfg['data']['samples_per_gpu']

    # Loss 정보
    # rpn_cls_loss = cfg['model']['rpn_head']['loss_cls']['type']
    # rpn_bbox_loss = cfg['model']['rpn_head']['loss_bbox']['type']
    # roi_cls_loss = cfg['model']['roi_head']['bbox_head']['loss_cls']['type']
    # roi_bbox_loss = cfg['model']['roi_head']['bbox_head']['loss_bbox']['type']

    # Optimizer 정보
    optimizer = cfg['optimizer']['type']

    # Learning rate
    learning_rate = cfg['optimizer']['lr']

    # Epoch
    epochs = cfg['runner']['max_epochs']

    #=================================================================================================================#
    # train log loader

    # 파일을 열고 마지막 줄만 읽어와서 딕셔너리로 변환
    with open(log_file_path, 'r') as file:
        last_line = file.readlines()[-1].strip()  # 마지막 줄을 읽고 공백 제거

    # JSON 문자열을 딕셔너리로 변환
    train_log = json.loads(last_line)

    last_lr = train_log['lr']
    loss_cls = train_log['loss_cls']
    loss_bbox = train_log['loss_bbox']
    loss_centerness = train_log['loss_centerness']
    total_loss = train_log['loss']
    #=================================================================================================================#

    params = []
    params.append(samples_per_gpu)
    # params.append(rpn_cls_loss)
    # params.append(rpn_bbox_loss)
    # params.append(roi_cls_loss)
    # params.append(roi_bbox_loss)
    params.append(optimizer)
    params.append(learning_rate)
    params.append(epochs)
    params.append(last_lr)
    params.append(loss_cls)
    params.append(loss_bbox)
    params.append(loss_centerness)
    params.append(total_loss)

    cols = []

    cols.append('Samples/GPU')
    # cols.append('RPN Cls Loss')
    # cols.append('RPN BBox Loss')
    # cols.append('RoI Cls Loss')
    # cols.append('RoI BBox Loss')
    cols.append('Optimizer')
    cols.append('Lr')
    cols.append('epochs')
    cols.append('last_lr')
    cols.append('loss_cls')
    cols.append('loss_bbox')
    cols.append('loss_centerness')
    cols.append('total_loss')
    
    
    try:
        # 워크시트가 있는지 확인
        worksheet = doc.worksheet(worksheet_name)
    except WorksheetNotFound:
        # 워크시트가 없으면 새로 생성
        worksheet = doc.add_worksheet(title=worksheet_name, rows="1000", cols="30")
        worksheet.append_rows([cols])

        print(f"'{worksheet_name}' 워크시트가 생성되었습니다.")


    worksheet = doc.worksheet(cfg['model']['type'])
    worksheet.append_rows([params])