import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import utils
import os
import imageio
import time

# def dx_dy_dt(img1,img2,u,v):
#     height=len(img1)
#     width=len(img1[0])
#     dx=np.zeros_like(img1)
#     dy=np.zeros_like(img1)
#     dt=np.zeros_like(img1)
#
#     for ii in range(height):
#         for jj in range(width):
#             # i_=int(ii+u[ii][jj])
#             # j_=int(jj+v[ii][jj])
#             i_=ii
#             j_=jj
#             i_=np.clip(i_,0,height-2)
#             j_=np.clip(j_,0,width-2)
#             i=np.clip(ii,0,height-2)
#             j=np.clip(jj,0,width-2)
#             dx[ii][jj]=0.25*(img1[i+1][j]-img1[i][j]+img1[i+1][j+1]-img1[i][j+1]+img2[i_+1][j_]-img2[i_][j_]+img2[i_+1][j_+1]-img2[i_][j_+1])
#             dy[ii][jj]=0.25*(img1[i][j+1]-img1[i][j]+img1[i+1][j+1]-img1[i+1][j]+img2[i_][j_+1]-img2[i_][j_]+img2[i_+1][j_+1]-img2[i_+1][j_])
#             dt[ii][jj]=0.25*(img2[i_][j_]-img1[i][j]+img2[i_+1][j_]-img1[i+1][j]+img2[i_][j_+1]-img1[i][j+1]+img2[i_+1][j_+1]-img1[i+1][j+1])
#     return dx,dy,dt
def dx_dy_dt(img1,img2,u,v):
    kernel_x=np.array([[-1,-1],[1,1]])
    kernel_y=np.array([[-1,1],[-1,1]])
    kernel_t=np.array([[1,1],[1,1]])
    dx=(cv2.filter2D(img1,-1,kernel_x)+cv2.filter2D(img2,-1,kernel_x))*0.25
    dy=(cv2.filter2D(img1,-1,kernel_y)+cv2.filter2D(img2,-1,kernel_y))*0.25
    dt=(cv2.filter2D(img2,-1,kernel_t)-cv2.filter2D(img1,-1,kernel_t))*0.25
    return dx,dy,dt

def clip_u_v(height,width,u,v):
    for ii in range(height):
        for jj in range(width):
            u[ii][jj]=np.clip(u[ii][jj],-ii,height-1-ii)
            v[ii][jj]=np.clip(v[ii][jj],-jj,width-1-jj)

def optical_flow(img1,img2,u,v,a,it_num):
    height=len(img1)
    width=len(img1[0])
    img1=cv2.GaussianBlur(img1,(5,5),0)
    img2=cv2.GaussianBlur(img2,(5,5),0)
    avg_kernel=np.array([[1/12,1/6,1/12],[1/6,0,1/6],[1/12,1/6,1/12]])
    a2=a*a
    dx,dy,dt=dx_dy_dt(img1,img2,u,v)
    for i in range(it_num):
        clip_u_v(height,width,u,v)
        u_=cv2.filter2D(u,-1,avg_kernel)
        v_=cv2.filter2D(v,-1,avg_kernel)
        b=dx*u_+dy*v_+dt
        c=a2+dx*dx+dy*dy
        u=u_-dx*b/c
        v=v_-dy*b/c
    return u,v

def HS_pyramid_optical_flow(img1,img2,u,v,a,it_num,level_num):
    width=img1.shape[1]
    height=img1.shape[0]
    for i in range(level_num,0,-1):
        scale=math.pow(2,i-1)
        new_shape=(int(width/scale),int(height/scale))
        img1i=cv2.resize(img1,new_shape)
        img2i=cv2.resize(img2,new_shape)
        u=cv2.resize(u,new_shape)*2
        v=cv2.resize(v,new_shape)*2
        u,v=optical_flow(img1i,img2i,u,v,a,it_num)
    return u,v

def run():
    output_folder='./frames/Horn/'
    input_folder='./frames/cut/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    imgs=[]
    for i in range(30):
        img1 = utils.read_img(os.path.join(input_folder,'./{:d}.jpg'.format(i)))
        img2 = utils.read_img(os.path.join(input_folder,'./{:d}.jpg'.format(i+1)))
        u=np.zeros_like(img1)
        v=np.zeros_like(img2)
        a=2
        it_num=10
        u,v=optical_flow(img1,img2,u,v,2,10)
        #u,v=HS_pyramid_optical_flow(img1,img2,u,v,a,3,4)
        flow_len=np.sqrt(u*u+v*v)
        #plt.imshow(flow_len)
        output_img=os.path.join(output_folder,'./{:d}.jpg'.format(i))
        plt.imsave(output_img,flow_len)
        imgs.append(cv2.imread(output_img))

    imageio.mimsave('./frames/Horn_pyramid/Horn.gif',imgs,duration=1)


if __name__=='__main__':
    # img1=utils.read_img('./2.jpg')
    # img2=utils.read_img('./3.jpg')
    # u=np.zeros_like(img1)
    # v=np.zeros_like(img1)
    # a=2
    # it_num=3
    # u,v=HS_pyramid_optical_flow(img1,img2,u,v,a,3,4)
    # utils.draw_dense_flow(img1,u,v,'./output/flow_Horn.jpg')
    run()
