import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('logfile', help='log file path')

    # 인자 받기
    args = parser.parse_args()

    with open(args.logfile, "r") as f:
        data = [json.loads(line) for line in f]
    
    train_logs = [log for log in data if log['mode'] == 'train']
    val_logs = [log for log in data if log['mode'] == 'val']

    folder_path = os.path.split(args.logfile)[0]

    # train log 파일 생성
    with open(os.path.join(folder_path, "train_logs.json"), "w") as f:
        for log in train_logs:
            json.dump(log, f)
            f.write("\n")

    # val log 파일 생성
    with open(os.path.join(folder_path, "val_logs.json"), "w") as f:
        for log in val_logs:
            json.dump(log, f)
            f.write("\n")
            
    print("train_log.json과 val_log.json 파일로 나누어 저장")

    if os.path.exists(args.logfile):
        confirm = input(f"{args.logfile}(통합 log) 파일을 삭제하시겠습니까?? (y/n): ").lower()

        if confirm == 'y':
            os.remove(args.logfile)
            print(f"{args.logfile} 파일 삭제 완료")
        else :
            print(f"{args.logfile} 파일 삭제 취소")
    else :
        print(f"{args.logfile} 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    main()