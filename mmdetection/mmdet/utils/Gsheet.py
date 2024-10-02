import gspread
from gspread.exceptions import WorksheetNotFound

# json 파일이 위치한 경로를 값으로 줘야 합니다.
def Gsheet_param(cfg):
    json_file_path = "/data/ephemeral/level2-objectdetection-cv-14-bde074188a7e.json"
    gc = gspread.service_account(json_file_path)
    spreadsheet_url = "https://docs.google.com/spreadsheets/d/1ElPQJ4cm0pka6ilYtxyynCzT80zLOryZhOnHDH2dWIc/edit?gid=0#gid=0"
    doc = gc.open_by_url(spreadsheet_url)
    # model type
    worksheet_name = cfg['model']['type']
    
    # Samples per GPU
    samples_per_gpu = cfg['data']['samples_per_gpu']

    # Base batch size for auto learning rate scaling
    base_batch_size = cfg['auto_scale_lr']['base_batch_size']

    # Loss 정보
    rpn_cls_loss = cfg['model']['rpn_head']['loss_cls']['type']
    rpn_bbox_loss = cfg['model']['rpn_head']['loss_bbox']['type']
    roi_cls_loss = cfg['model']['roi_head']['bbox_head']['loss_cls']['type']
    roi_bbox_loss = cfg['model']['roi_head']['bbox_head']['loss_bbox']['type']

    # Optimizer 정보
    optimizer = cfg['optimizer']['type']

    # Learning rate
    learning_rate = cfg['optimizer']['lr']

    # Epoch
    epochs = cfg['runner']['max_epochs']

    # 결과 출력
    print('Model type:', worksheet_name)
    print("Samples per GPU:", samples_per_gpu)
    print("Base Batch Size (for auto LR scaling):", base_batch_size)
    print("RPN Classification Loss:", rpn_cls_loss)
    print("RPN Bounding Box Loss:", rpn_bbox_loss)
    print("ROI Classification Loss:", roi_cls_loss)
    print("ROI Bounding Box Loss:", roi_bbox_loss)
    print("Optimizer:", optimizer)
    print("Learning Rate:", learning_rate)
    print("Epochs:", epochs)

    params = []
    params.append(samples_per_gpu)
    params.append(base_batch_size)
    params.append(rpn_cls_loss)
    params.append(rpn_bbox_loss)
    params.append(roi_cls_loss)
    params.append(roi_bbox_loss)
    params.append(optimizer)
    params.append(learning_rate)
    params.append(epochs)

    cols = []

    cols.append('Samples/GPU')
    cols.append('Batch Size')
    cols.append('RPN Cls Loss')
    cols.append('RPN BBox Loss')
    cols.append('RoI Cls Loss')
    cols.append('RoI BBox Loss')
    cols.append('Optimizer')
    cols.append('Lr')
    cols.append('epochs')
    
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