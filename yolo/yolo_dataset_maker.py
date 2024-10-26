import json
import os
import shutil

origin_image_dir = './datasets/origin_dataset/train'

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
            x,y,w,h = item['bbox'][0], item['bbox'][1], item['bbox'][2], item['bbox'][3]
            centerx = x + w / 2.
            centery = y + h / 2.
            centerx /= 1024.
            centery /= 1024.
            w /= 1024.
            h /= 1024.
            x_center = "{:.6f}".format(centerx)
            y_center = "{:.6f}".format(centery)
            width = "{:.6f}".format(w)
            height = "{:.6f}".format(h)
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


for i in range(1, 6):
    train_fold_json_path = f'./datasets/sgkf_5_14/train_{i}fold.json'
    val_fold_json_path = f'./datasets/sgkf_5_14/val_{i}fold.json'
    train_output_dir = f'./datasets/recycle_trash_dataset{i}/train'
    val_output_dir = f'./datasets/recycle_trash_dataset{i}/valid'

    dataset_maker(train_fold_json_path, train_output_dir)
    dataset_maker(val_fold_json_path, val_output_dir)

        
    
        
