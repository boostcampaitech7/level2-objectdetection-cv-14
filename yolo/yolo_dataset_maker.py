import json
import os
import shutil

origin_image_dir = './datasets/origin_dataset/train'

train_fold_json_path = './datasets/sgkf_5_14/train_1fold.json'
val_fold_json_path = './datasets/sgkf_5_14/val_1fold.json'
train_output_dir = './datasets/recycle_trash_dataset/train'
val_output_dir = './datasets/recycle_trash_dataset/valid'


def dataset_maker(read_path, target_path):
    with open(read_path, 'r') as f:
        train_ann_data = json.load(f)        
        id = 0
        content = ''
        for idx, item in enumerate(train_ann_data['annotations']):
            if item['image_id'] != id:
                filename = f"{id:04d}.txt"
                output_path = os.path.join(target_path+'/labels', filename)
                with open(output_path, 'w') as ff:
                    print(output_path)
                    ff.write(content)
                imgname = f"{id:04d}.jpg"
                src_path = os.path.join(origin_image_dir, imgname)
                dst_path = os.path.join(target_path+'/images', imgname)
                shutil.copy2(src_path, dst_path)
                id = item['image_id']
                content = ''                    

            category_id = str(item['category_id'])
            x_center = "{:.6f}".format(((item['bbox'][0]*2 + item['bbox'][2])/2.0) / 1024)
            y_center = "{:.6f}".format(((item['bbox'][1]*2 + item['bbox'][3])/2.0) / 1024)
            width = "{:.6f}".format(item['bbox'][2] / 1024)
            height = "{:.6f}".format(item['bbox'][3] / 1024)
            data = category_id + " " + x_center + " " + y_center + " " + width + " " + height + "\n"
            content += data
    
        filename = f"{id:04d}.txt"
        output_path = os.path.join(target_path+'/labels', filename)
        with open(output_path, 'w') as ff:
            print(output_path)
            ff.write(content)
        imgname = f"{id:04d}.jpg"
        src_path = os.path.join(origin_image_dir, imgname)
        dst_path = os.path.join(target_path+'/images', imgname)
        shutil.copy2(src_path, dst_path)




dataset_maker(train_fold_json_path, train_output_dir)
dataset_maker(val_fold_json_path, val_output_dir)

        
    
        
