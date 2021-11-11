#import pandas as pd
import os
import json
root_dir = './json/'

json_list = os.listdir(root_dir)
json_list = [file for file in json_list if file.endswith(".json")]

img_pd_list = []
anno_pd_list = []
cate_pd_list = []

for idx, cur_json in enumerate(json_list):
    with open(root_dir+cur_json, 'r', encoding="utf-8") as f:
        data = json.load(f)
    
    print(cur_json)

    tmp_pd = pd.DataFrame(data['images'])
    tmp_pd['json_idx'] = idx
    tmp_pd['is_eval'] = False
    is_eval = tmp_pd['id'] %5 == 0 # 각 super 카테별 20%만 eval
    tmp_pd.loc[is_eval, ["is_eval"]] = True
    img_pd_list.append(tmp_pd)
    print('imgs: ', len(tmp_pd))

    tmp_pd = pd.DataFrame(data['annotations'])
    tmp_pd['json_idx'] = idx
    anno_pd_list.append(tmp_pd)
    print('anno: ',len(tmp_pd))

    tmp_pd = pd.DataFrame(data['categories'])
    tmp_pd['json_idx'] = idx
    cate_pd_list.append(tmp_pd)
    print('cls: ',len(tmp_pd))

    print()
    del tmp_pd

total_imgs = pd.concat(img_pd_list)
total_imgs.reset_index(inplace=True, drop=True) # 통합 idx
total_imgs.drop_duplicates('file_name', inplace=True, keep='first') # 파일 중복 검사

total_anno = pd.concat(anno_pd_list)
total_anno.reset_index(inplace=True, drop=True)
total_anno['train_cls_idx'] = -1
total_anno['train_img_idx'] = -1
print(len(total_anno))

total_cate = pd.concat(cate_pd_list)
total_cate.reset_index(inplace=True, drop=True) # 통합 idx

# iterrows() 느림
for idx, val in total_cate.iterrows():
    cur = (total_anno.json_idx == val['json_idx']) & (total_anno.category_id == val['id'])
    total_anno.loc[cur, 'train_cls_idx'] = idx

for idx, val in total_imgs.iterrows():
    cur = (total_anno.json_idx == val['json_idx']) & (total_anno.image_id == val['id'])
    total_anno.loc[cur, 'train_img_idx'] = idx

print(len(total_anno))
total_anno.to_csv('total_anno.csv')
total_imgs.to_csv('total_imgs.csv')
total_cate.to_csv('total_cate.csv')