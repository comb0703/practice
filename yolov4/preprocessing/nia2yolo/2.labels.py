import pandas as pd

# read images info, annotation
total_imgs = pd.read_csv('/content/total_imgs.csv')
total_anno = pd.read_csv('/content/nia_total.csv')

# dir
eval_label_dir = '/labels/eval/'
train_label_dir = '/labels/train/'

for idx, val in total_imgs.iterrows():
    # for eval
  if val['is_eval'] == True :
      txt_path = eval_label_dir+val['file_name'][:-3]+"txt"
      with open(txt_path, 'w') as f:
          tmp_idx = total_anno['train_img_idx'] == idx
          cur_frame = total_anno.loc[tmp_idx]
          
          for c_idx, c_val in cur_frame.iterrows():
              # real_cate,yolo_cx,yolo_cy,yolo_w,yolo_h
              ann = '{} {} {} {} {}\n'.format(
                  int(c_val['train_cls_idx']), 
                  round(c_val['yolo_cx'],5),round(c_val['yolo_cy'],5), round(c_val['yolo_w'],5), round(c_val['yolo_h'],5))
              f.write(ann)  # label format
    
    # for train
  else :
      txt_path = train_label_dir+val['file_name'][:-3]+"txt"
      with open(txt_path, 'w') as f:
          tmp_idx = total_anno['train_img_idx'] == idx
          cur_frame = total_anno.loc[tmp_idx]
          
          for c_idx, c_val in cur_frame.iterrows():
              # real_cate,yolo_cx,yolo_cy,yolo_w,yolo_h
              ann = '{} {} {} {} {}\n'.format(
                  int(c_val['train_cls_idx']), 
                  round(c_val['yolo_cx'],5),round(c_val['yolo_cy'],5), round(c_val['yolo_w'],5), round(c_val['yolo_h'],5))
              f.write(ann)  # label format