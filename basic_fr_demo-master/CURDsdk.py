import os
import sys
import cv2
import numpy as np

from CURD_css import CURD_color
from RetinaFace.retinaface import RetinaFace_mgr

import torch
from skimage import transform as trans
import pandas as pd

class CURD_sdk:

    def __init__(self):

        self.init_done = False
        self.detector = None

        self.reid_thres = 35
        self.model_path = 'models/Glint360k_r18.pth'
        
        self.gal_root = 'gal/'
        self.gal_file_name = 'Glint360k_r18'
        
        # draw cv2 color
        self.normal_box_color = CURD_color['normal_box']
        self.not_work_box_color = CURD_color['not_box']
        self.white = CURD_color['white']
        self.navy = CURD_color['navy']

        # config (fixed)
        self.device = 'cuda'
        self.src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041] ], dtype=np.float32 )        
        self.tform = trans.SimilarityTransform()

    def init_sdk(self):
        try:
            self.set_camera()
            self.set_detector()
            self.set_extractor()
            self.make_gal()
            self.init_done = True
        except Exception as e:
            sys.stdout.write(str(e))
            sys.stdout.flush()
            return False
        else:
            return True

    #### Setting
    def set_camera(self):
        self.cap = cv2.VideoCapture(0) # 0 laptop cam , 2 realsense color cam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    def set_detector(self):
        self.detector = RetinaFace_mgr()

    ### Extractor
    def set_extractor(self):
        self.extractor = self.get_extractor(self.model_path)
        self.extractor.to(self.device)
        self.extractor.eval()

    def get_extractor(self, model_path):
        sys.path.append('./models/arcface_torch')
        from iresnet import iresnet18
        net = iresnet18(fp16=False)
        net.load_state_dict(torch.load(model_path))
        sys.stdout.write('\t[Network] %s\n\n' % model_path)
        sys.stdout.flush()
        return net

    def extract_emb(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.transpose(face, (2, 0, 1))
        face = torch.from_numpy(face).unsqueeze(0).float()
        face.div_(255).sub_(0.5).div_(0.5)
        
        self.extractor.eval()
        return self.extractor(face.to(self.device)).cpu().detach().numpy()

    ### get a frame from camera
    def get_rgb_frame(self):    
        ret, frame = self.cap.read()
        if not ret:
            return None
        else:
            return frame

    ### gallery & matching
    def make_gal(self):
        file_list = os.listdir(self.gal_root)
        
        gal_csv = [ file for file in file_list if file.endswith('.csv') ]
        gal_npy = [ file for file in file_list if file.endswith('.npy') ]
        file_list = [ file for file in file_list if file.endswith('.jpg') or file.endswith('.png') ]
        
        # 갤러리 파일이 있다면 (.csv, .npy)
        if len(gal_csv)  == 1 and len(gal_npy) == 1:
            self.gal_list = pd.read_csv(self.gal_root+gal_csv[0], index_col=0)
            self.gal_embs = torch.from_numpy(np.load(self.gal_root+gal_npy[0]))
        
        # 갤러리 파일을 생성해야 한다면
        else:
            self.gal_list = pd.DataFrame(file_list, columns=['file'])
            self.gal_list['emb'] = False
            self.gal_embs = np.zeros(shape=(len(self.gal_list), 512), dtype=np.float32)

            for idx, val in self.gal_list.iterrows():
                face_file_name = val['file']
                img = cv2.imread(self.gal_root+face_file_name)

                lm = self.get_bbox(img.copy(), gal=True)
                if len(lm) > 0:
                    face = self.get_face(img, lm)
                    self.gal_embs[idx] = self.extract_emb(face)
                    self.gal_list.loc[idx, ['emb']] = True
                else:
                    self.gal_list.loc[idx, ['emb']] = False
                    continue
                print(face_file_name)

            self.gal_list.to_csv(self.gal_root+self.gal_file_name+'.csv')
            np.save(self.gal_root+self.gal_file_name, self.gal_embs)
            self.gal_embs = torch.from_numpy(self.gal_embs)

        print("==> gallery task done | imgs : ", len(self.gal_embs))
        print()

    def calculate_score(self, prob, reg):
        
        x = torch.squeeze(prob).to(self.device)
        xx = torch.dot(x, x)

        y = reg.to(self.device)
        yy = torch.pow(y, 2).to(self.device)
        yy = torch.sum(yy, dim=1)
        
        xy = torch.matmul(y, x)
        norm = torch.sqrt(xx*yy) + 0.000001

        cosine_dist = 1.0 - xy / norm # batch
        cosine_dist = torch.max(torch.min(cosine_dist, torch.ones([1], device=self.device)), torch.zeros([1], device=self.device)).cpu() # batch

        cosine_dist = torch.where(cosine_dist >= 1.0, torch.tensor(1.), cosine_dist) 
        score = (1.0 - cosine_dist) * 100

        return score
            
    def match_face(self, prob_face_emb):
        
        score = self.calculate_score(prob_face_emb, self.gal_embs)
        top1_idx = int(torch.argmax(score))
        top1_score = float(score[top1_idx])

        if top1_score > self.reid_thres :
            top1_name = self.gal_list.loc[top1_idx]['file'][:-4]
        else:
            top1_name = 'Unknown'
            top1_score = -1

        return top1_score, top1_name

    def get_bbox(self, frame, gal=False):

        if frame is not None:
            # Detect faces
            boxes = self.detector(frame, 'color')
            result_boxes = []
            lm = []
            num_faces = len(boxes)
            if num_faces == 0:
                if gal:
                    return lm
                else:
                    return num_faces, result_boxes, lm
            else:
                frame_h, frame_w, _ = frame.shape
                for box in boxes:
                    left, top, right, bottom = box[:4]

                    w = right - left
                    h = bottom - top

                    lm.append( [box[4:]] )
                    result_boxes.append( [left, top, right, bottom, int(w*h)] )

                if gal:
                    return lm[0]
                else:
                    return num_faces, result_boxes, lm

    def get_boxed_frame(self, cur_frame, boxes):

        for box in boxes:
            cur_rgb = self.normal_box_color

            box_left = box[0]
            box_top = box[1]
            box_right = box[2]
            box_btm = box[3]

            # face bbox
            cur_frame = cv2.rectangle(cur_frame, (box_left, box_top), (box_right, box_btm), (cur_rgb[0], cur_rgb[1], cur_rgb[2]), thickness=2)

        return cur_frame

    def get_face(self, frame, lm):

        lm = np.array(lm).astype(np.float32).reshape((5,2))
        self.tform.estimate(lm, self.src)
        M = self.tform.params[0:2,:]
        warped = cv2.warpAffine(frame, M, (112, 112), borderValue=0.0)

        return warped

    #### CALL from external ####
    def is_init_done(self):
        return self.init_done

    def get_frame(self, top_w, top_h):
        
        rgb_frame = self.get_rgb_frame()

        if rgb_frame is None:
            return np.zeros((top_w, top_h)), self.status

        rgb_frame = cv2.resize(rgb_frame, dsize=(top_w, top_h), interpolation=cv2.INTER_AREA)
        bbox_list = []

        num_faces, bbox_list, lm_list = self.get_bbox(rgb_frame.copy()) # rgb orginal
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        result_frame = rgb_frame.copy()

        if num_faces > 0:
            result_frame = self.get_boxed_frame(result_frame, bbox_list)

            for bbox, lm in zip(bbox_list, lm_list):
                prob_face = self.get_face(rgb_frame.copy(), lm)
                prob_face_emb = self.extract_emb(prob_face)
                pred_score, pred_name = self.match_face(torch.from_numpy(prob_face_emb))
                
                name_color = [110,110,110] # unknown color

                # 유효 점수
                if pred_score > 0:
                    result_frame = cv2.putText(result_frame, str(pred_score)[:5], (bbox[0],bbox[3]+62), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.7, thickness=1, color=self.normal_box_color)
                    name_color = [0,200,197]

                # 이름 배경
                result_frame = cv2.rectangle(result_frame, (bbox[0], bbox[3]+10), (bbox[2], bbox[3]+38), (255,255,255), thickness=-1)
                
                # 이름
                result_frame = cv2.putText(result_frame, pred_name, (bbox[0],bbox[3]+30), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.7, thickness=2, color=name_color)

        return result_frame