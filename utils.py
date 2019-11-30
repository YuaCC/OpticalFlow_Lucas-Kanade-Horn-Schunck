import cv2
import numpy as np
import matplotlib.pyplot as plt
__all__=['read_img']
def read_img(file:str):
    img=cv2.imread(file)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=np.array(img,np.float)
    return img

def draw_arrow(img,x0,y0,x1,y1):
    img=cv2.circle(img,(x0,y0),3,(0,255,0),-1)
    img=cv2.line(img,(x0,y0),(x1,y1),(0,255,0),1)
    return img

def draw_sparse_flow(img,points,colors,file):
    for i,point in enumerate(points):
        x,y=point[0]
        x,y=int(x),int(y)
        img=cv2.circle(img,(x,y),5,colors[i].tolist(),-1)
        cv2.imwrite(file,img)

def draw_dense_flow(img,flowx,flowy,file):
    arr_len=15
    mean=(np.mean(abs(flowx))+np.mean(abs(flowy)))/2
    arr_x = -flowx / (mean*4) * arr_len
    arr_y = -flowy / (mean*4) * arr_len
    arr_x = np.clip(arr_x, -arr_len, arr_len)
    arr_y = np.clip(arr_y, -arr_len, arr_len)
    for i in range(arr_len,img.shape[0],arr_len*2):
        for j in range(arr_len,img.shape[1],arr_len*2):
            if (arr_x[i][j]**2+arr_y[i][j]**2)**0.5>=arr_len/2:
                draw_arrow(img,j,i,int(j+arr_x[i][j]),int(i+arr_y[i][j]))
    cv2.imwrite(file,img)