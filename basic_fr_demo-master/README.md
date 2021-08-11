### CURD DEMO
연구용 Face Recognition demo 입니다.<br><br>

## requirements
PyQt5<br>
pandas<br>
torch<br>
skimage<br><br>

## 갤러리 저장 형식
- 갤러리 폴더 내의 파일 리스트는 pandas Dataframe으로 저장됩니다. (.csv)
- 갤러리 폴더 내의 파일들의 임베딩 벡터는 numpy로 저장됩니다. (.npy)

## customizing
- CURDsdk.py 에서 sdk를 init 하는 부분을 수정합니다.

```
class CURD_sdk:
    def __init__(self):

        self.init_done = False
        self.detector = None

        self.reid_thres = 35
        self.model_path = 'models/Glint360k_r18.pth'
        
        self.gal_root = 'gal/'
        self.gal_file_name = 'Glint360k_r18'
```

<br>

- 백본 교체가 필요하다면, 아래 3개의 함수를 수정합니다.

```
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
```

- UI는 pyQT이지만, sdk의 get_frame( )을 이용하여 프레임만 받아보게 구성할 수 있습니다.
<br> 
