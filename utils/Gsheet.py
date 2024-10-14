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

    # User 명
    param_dict['user'] = os.path.abspath(__file__).split("/")[4]

    # model type
    param_dict['model'] = cfg['model']['type']

    # backbone type
    param_dict['backbone'] = cfg['model']['backbone']['type']
    
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
            param_dict['img_scale'] = str(pipe['img_scale'])
            break

    # scheduler
    param_dict['scheduler'] = cfg['lr_config']['policy']
    param_dict['lr_step'] = str(cfg['lr_config']['step'])

    # Epoch
    param_dict['max_epochs'] = cfg['runner']['max_epochs']

    #=================================================================================================================#
    # train log loader

    # 파일을 열고 마지막 줄만 읽어와서 딕셔너리로 변환
    with open(log_file_path, 'r') as file:
        last_line = file.readlines()[-1].strip()  # 마지막 줄을 읽고 공백 제거

    # JSON 문자열을 딕셔너리로 변환
    train_log = json.loads(last_line)

    # log에서 추출 가능한 loss 중에서 공통적인 것들로 구성
    param_dict['loss_cls'] = train_log['loss_cls']
    param_dict['loss_bbox'] = train_log['loss_bbox']
    param_dict['loss'] = train_log['loss']
    #=================================================================================================================#

    # sheet에 추가하기 위해서 값들을 list로 저장
    params = [param_dict[k] for k in param_dict]

    # sheet가 없는 경우 Head Row를 구성하기 위해서 Col 명을 list로 저장
    cols = [k.capitalize() for k in param_dict]
    
    try:
        # 워크시트가 있는지 확인
        worksheet = doc.worksheet(param_dict['model'])
    except WorksheetNotFound:
        # 워크시트가 없으면 새로 생성
        worksheet = doc.add_worksheet(title=param_dict['model'], rows="1000", cols="30")
        # Col 명 추가
        worksheet.append_rows([cols])

        # Header Cell 서식 
        header_formatter = CellFormat(
            backgroundColor=Color(0.9, 0.9, 0.9),
            textFormat=TextFormat(bold=True, fontSize=15),
            horizontalAlignment='CENTER',
        )
        
        # Header의 서식을 적용할 범위
        header_range = f"A1:{chr(ord('A') + len(cols) - 1)}1"

        # Header 서식 적용
        format_cell_range(worksheet, header_range, header_formatter)

        # Header Cell의 넓이 조정
        for idx, header in enumerate(cols):
            column_letter = chr(ord('A') + idx) 
            if idx == 1:
                header = param_dict['model']
            elif idx == 2:
                header = param_dict['backbone']
            width = max((len(header) + 4) * 10, 80)
            set_column_width(worksheet, column_letter, width)

        print(f"'{param_dict['model']}' 워크시트가 생성되었습니다.")

    # 실험 인자를 작성한 worksheet
    worksheet = doc.worksheet(cfg['model']['type'])

    # 실험 인자 worksheet에 추가
    worksheet.append_rows([params])

    # 현재 작성하는 실험 인자들 Cell의 서식
    # 노란색으로 하이라이트
    row_formatter = CellFormat(
        backgroundColor=Color(1, 1, 0),
        textFormat=TextFormat(fontSize=12),
        horizontalAlignment="CENTER"
    )

    # 이전 작성 실험인자들 배경색 원상복구
    rollback_formatter = CellFormat(
        backgroundColor=Color(1.0, 1.0, 1.0)
    )
    
    # 마지막 줄에만 하이라이팅이 들어가야 하므로 마지막 row 저장
    last_row = len(worksheet.get_all_values())
    row_range = f"A{last_row}:{chr(ord('A') + len(cols) - 1)}{last_row}"
    rollback_range = f"A{last_row - 1}:{chr(ord('A') + len(cols) - 1)}{last_row - 1}"
    
    # 헤더셀의 서식이 초기화되는 것을 방지하기 위한 조건문
    if last_row - 1 != 1:
        format_cell_range(worksheet, rollback_range, rollback_formatter)
    
    format_cell_range(worksheet, row_range, row_formatter)
