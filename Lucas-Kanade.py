import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import utils
import os
import imageio

def dx(img):
    kernel=np.array([[-0.5,0,0.5]])
    img_dx=cv2.filter2D(img,-1,kernel)
    img_dx[:,0]=img[:,1]-img[:,0]
    img_dx[:,-1]=img[:,-1]-img[:,-2]
    return img_dx

def dy(img):
    kernel=np.array([[-0.5],[0],[0.5]])
    img_dy=cv2.filter2D(img,-1,kernel)
    img_dy[0,:]=img[1,:]-img[0,:]
    img_dy[-1,:]=img[-1,:]-img[-2,:]
    return img_dy


def optical_flow(img1,img2,points,gx,gy,half_winsize,it_num):
    win_size=half_winsize*2+1
    img1=cv2.GaussianBlur(img1,(win_size,win_size),0)
    img2=cv2.GaussianBlur(img2,(win_size,win_size),0)
    height = img1.shape[0]
    width = img1.shape[1]
    ix=dx(img2)
    iy=dy(img2)
    for i,point in enumerate(points):
        x, y=point[0]

        for j in range(it_num):
            x_,y_=int(x+gx[i]),int(y+gy[i])
            if not(0<=x_<width and 0<=y_<height):
                print('points',i,'loss')
                continue
            l=int(max(0,min([x_,x,half_winsize])))
            r=int(max(0,min([width-1-x_,width-1-x,half_winsize])))
            u=int(max(0,min([y_,y,half_winsize])))
            d=int(max(0,min([height-1-y_,height-1-y,half_winsize])))

            ix_local=ix[y_-u:y_+d,x_-l:x_+r]
            iy_local=iy[y_-u:y_+d,x_-l:x_+r]
            try:
                img1_local=img1[int(y)-u:int(y)+d,int(x)-l:int(x)+r]
                img2_local=img2[y_-u:y_+d,x_-l:x_+r]
                it_local=img2_local-img1_local
                ixix_sum=np.sum(ix_local*ix_local)
                ixiy_sum=np.sum(ix_local*iy_local)
                iyiy_sum=np.sum(iy_local*iy_local)
                ixit_sum=np.sum(ix_local*it_local)
                iyit_sum=np.sum(iy_local*it_local)
                g=np.array([[ixix_sum,ixiy_sum],[ixiy_sum,iyiy_sum]])
                g_inv=np.linalg.inv(g)
                b=[[-ixit_sum],[-iyit_sum]]
                u=g_inv[0][0]*b[0][0]+g_inv[0][1]*b[1][0]
                v=g_inv[1][0]*b[0][0]+g_inv[1][1]*b[1][0]
                gx[i]+=u
                gy[i]+=v
            except Exception as e:
                print(e)


def LK_pyramid_optical_flow(img1,img2,points,it_num,level_num):
    gx=np.zeros((len(points)),np.float)
    gy=np.zeros((len(points)),np.float)
    width=img1.shape[1]
    height=img1.shape[0]
    for i in range(level_num,0,-1):
        scale=math.pow(2,i-1)
        new_shape=(int(width/scale),int(height/scale))
        img1i=cv2.resize(img1,new_shape)
        img2i=cv2.resize(img2,new_shape)
        gx=gx*2
        gy=gy*2
        optical_flow(img1i,img2i,points/scale,gx,gy,5,it_num)
    return gx,gy


if __name__=='__main__':
    if not os.path.exists('./frames/lucas'):
        os.mkdir('./frames/lucas')

    img_old=cv2.imread('./frames/cut/0.jpg')
    img_old_gray = cv2.cvtColor(img_old, cv2.COLOR_BGR2GRAY)
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    p0 = cv2.goodFeaturesToTrack(img_old_gray, mask=None, **feature_params)
    colors=np.random.randint(0,255,(100,3))
    img_old_gray=np.array(img_old_gray,np.float)
    imgs=[]
    for i in range(30):
        img_out='./frames/lucas/{:d}.jpg'.format(i)
        utils.draw_sparse_flow(img_old,p0,colors,img_out)
        imgs.append(imageio.imread(img_out))
        img_new = cv2.imread('./frames/cut/{:d}.jpg'.format(i+1))
        img_new_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
        img_new_gray=np.array(img_new_gray,np.float)
        gx,gy=LK_pyramid_optical_flow(img_old_gray,img_new_gray,p0,3,4)
        p0[:,0,0]+=gx
        p0[:,0,1]+=gy
        img_old=img_new
        img_old_gray=img_new_gray
    imageio.mimsave('./frames/lucas/lucas.gif',imgs,duration=1)
