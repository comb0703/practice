import pandas as pd

# read annotation
df2 = pd.read_csv('/content/total_anno.csv',index_col=0)

# add x,y,w,h from bbox
tmp_pd = pd.DataFrame(df2['bbox'].str[1:-1].str.split(", ").tolist(), columns=['x','y','w','h'])
df2 = pd.merge(df2, tmp_pd, left_index=True, right_index=True)
df2[['x','y','w','h']] = df2[['x','y','w','h']].astype(float)

# add height, width
df2['height'] = 2100.0
df2['width'] = 2800.0

def nia2yolo(img_w, img_h, x, y, bw, bh):

    dw = 1. / img_w
    dh = 1. / img_h

    x = x if x >= 0 else 0
    y = y if y >= 0 else 0

    x = x if x <= img_w else img_w
    y = y if y <= img_h else img_h

    w = bw
    h = bh
    x = x + (w/2.0)
    y = y + (h/2.0)

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return [round(x, 5), round(y, 5), round(w, 5), round(h, 5)]

# add yolo bbox
df2[['yolo_cx','yolo_cy','yolo_w','yolo_h']] = df2.apply(lambda row: nia2yolo(row['width'],row['height'],row['x'],row['y'],row['w'],row['h']), axis=1).tolist()

# 학습에 이용할 bbox 범위 설정
query = (df2['w'] <= 200.0) & (df2['h'] <= 200.0)
df2 = df2[query]

# 학습에 이용할 bbox 범위 설정
# query = df2['area'] <= 40000.0
# df2 = df2[query]

# save file
df2.to_csv('nia_total.csv')