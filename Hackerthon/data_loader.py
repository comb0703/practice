import os
import json
from nsml import DATASET_PATH

def test_data_loader(root_path):
    data_path = os.path.join(root_path, 'test', 'test_data')
    file_list = os.listdir(data_path)
    imgs_path = [os.path.join(data_path, file_name) for file_name in file_list]
    return imgs_path

def feed_infer(output_file, infer_func): 
    # result 형식은 {'file_name':[ [], [] ], 'file_name':[ [], [] ]}

    imgs_path = test_data_loader(DATASET_PATH)
    pred_dict = infer_func(imgs_path)
    
    with open(output_file, 'w') as f:
        json.dump(pred_dict, f)