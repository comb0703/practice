import pandas as pd

# read images info
images = pd.read_csv("total_imgs.csv", index_col=0)

# seperate train,eval
q1 = 'is_eval == True'
q2 = 'is_eval == False'
images_eval = images.query(q1)
images_train = images.query(q2)

train_path = 'train.txt'
eval_path = 'eval.txt'

img_train_dir = './images/train/'
img_eval_dir = './images/eval/'

with open(eval_path, 'w') as f:
  for i in images_eval['file_name'] :
    f.write(img_eval_dir+i+'\n')
    
with open(train_path, 'w') as f:
  for i in images_train['file_name'] :
    f.write(img_train_dir+i+'\n')
    