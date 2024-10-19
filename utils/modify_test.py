import json


# pseudo_labels.json 파일의 annotations 부분에서 image_id 수정 후 저장하는 코드
def modify_pseudo_labels(input_file, output_file):
    # JSON 파일 읽기
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    # image_id 필드 수정 (파일명에서 숫자 부분에 10000 더하기)
    if "annotations" in data:
        for annotation in data["annotations"]:
            image_id_str = annotation["image_id"].split("/")[-1].split(".")[0]
            new_image_id = str(int(image_id_str) + 10000)
            annotation["image_id"] = int(new_image_id)
    
    # 수정된 데이터를 새로운 JSON 파일에 저장
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)



# pseudo_labels.json 파일 수정 호출
pseudo_input_file = '/data/ephemeral/home/dataset/csv_pseudo.json'
pseudo_output_file = '/data/ephemeral/home/dataset/csv_pseudo_modified.json'
modify_pseudo_labels(pseudo_input_file, pseudo_output_file)