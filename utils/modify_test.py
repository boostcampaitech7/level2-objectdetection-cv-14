import json
import argparse
#test annotation의 id와 train annotation의 id가 겹치는 것을 방지하기 위해 10000을 더해주기 

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



if __name__ == "__main__":
    # Argument parser 설정
    parser = argparse.ArgumentParser(description="Modify image_id in pseudo_labels JSON file.")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file")
    parser.add_argument("output_file", type=str, help="Path to save the modified JSON file")

    args = parser.parse_args()

    # 파일 경로를 인자로 받아서 함수 호출
    modify_pseudo_labels(args.input_file, args.output_file)