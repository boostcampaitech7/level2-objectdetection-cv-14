import os
import json
import gspread
from gspread.exceptions import WorksheetNotFound
from gspread_formatting import *
from dotenv import dotenv_values

# json 파일이 위치한 경로를 값으로 줘야 합니다.
def Gsheet_param(cfg, output_dir):
    # env 파일 불러오기
    env_path = "/data/ephemeral/home/dataset/.env"
    env = dotenv_values(env_path)

    # train log 파일 경로 설정
    log_file_path = os.path.join(output_dir, "None.log.json")

    # 서비스 연결
    gc = gspread.service_account(env['JSON_PATH'])

    # url에 따른 spread sheet 열기
    doc = gc.open_by_url(env['URL'])

    # 저장할 변수 dict 선언
    param_dict = dict()
    # model type
    param_dict['worksheet_name'] = cfg['model']['type']
    
    # Samples per GPU
    param_dict['samples_per_gpu'] = cfg['data']['samples_per_gpu']

    # Optimizer 정보
    param_dict['optimizer'] = cfg['optimizer']['type']

    # Learning data
    param_dict['lr'] = cfg['optimizer']['lr']
    param_dict['weight_decay'] = cfg['optimizer']['weight_decay']

    # train시 image scale
    for pipe in cfg['train_pipeline']:
        if pipe['type'] == "Resize":
            param_dict['img_scale'] = pipe['img_scale']
            break

    # scheduler
    param_dict['scheduler'] = cfg['lr_config']['policy']
    param_dict['lr_step'] = cfg['lr_config']['step']

    # Epoch
    param_dict['max_epochs'] = cfg['runner']['max_epochs']

    #=================================================================================================================#
    # train log loader

    # 파일을 열고 마지막 줄만 읽어와서 딕셔너리로 변환
    with open(log_file_path, 'r') as file:
        last_line = file.readlines()[-1].strip()  # 마지막 줄을 읽고 공백 제거

    # JSON 문자열을 딕셔너리로 변환
    train_log = json.loads(last_line)

    param_dict['loss_cls'] = train_log['loss_cls']
    param_dict['loss_bbox'] = train_log['loss_bbox']
    param_dict['loss'] = train_log['loss']
    #=================================================================================================================#

    params = [param_dict[k] for k in param_dict]

    cols = [k.capitalize() for k in param_dict]
    cols[0] = 'Model'
    
    try:
        # 워크시트가 있는지 확인
        worksheet = doc.worksheet(param_dict['worksheet_name'])
    except WorksheetNotFound:
        # 워크시트가 없으면 새로 생성
        worksheet = doc.add_worksheet(title=param_dict['worksheet_name'], rows="1000", cols="30")
        worksheet.append_rows([cols])

        header_formatter = CellFormat(
            backgroundColor=Color(0.9, 0.9, 0.9),
            textFormat=TextFormat(bold=True, fontSize=12),
            horizontalAlignment='CENTER',
        )
        
        header_range = f"A1:{chr(ord('A') + len(cols) - 1)}1"

        format_cell_range(worksheet, header_range, header_formatter)

        for idx, header in enumerate(cols):
            column_letter = chr(ord('A') + idx) 
            if idx == 0:
                header = param_dict['worksheet_name']
            width = max((len(header) + 2) * 10, 70)
            set_column_width(worksheet, column_letter, width)

        print(f"'{param_dict['worksheet_name']}' 워크시트가 생성되었습니다.")

    worksheet = doc.worksheet(cfg['model']['type'])
    worksheet.append_rows([params])
