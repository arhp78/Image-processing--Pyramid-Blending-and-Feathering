# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 10:46:04 2021

@author: hatam
"""

import numpy as np
import cv2

lengh=735
witdth=324
def make_laplacian(target,source,mask,kernel):
    blured_target=cv2.filter2D(target, -1, kernel)   
    laplacian_target=target-blured_target
    
    blured_source=cv2.filter2D(source, -1, kernel)   
    laplacian_source=source-blured_source
    
    blured_mask=cv2.filter2D(mask, -1, kernel)   
    
    
    return laplacian_target,blured_target ,laplacian_source,blured_source,blured_mask
#read image
target=cv2.imread("2.target.jpg")
source=cv2.imread("1.source.jpg")

#creat mask source
mask_source=np.zeros_like(target)
mask_source[lengh:lengh+len(source),witdth:witdth+len(source[0]),:]=source

#creat binery mask
source1=cv2.imread("mask-source.jpg")
mask=np.zeros_like(target)
mask[lengh:lengh+len(source),witdth:witdth+len(source[0]),:]=source1
x,y=np.where(mask[:,:,0]>=10)
x1,y1=np.where(mask[:,:,0]<10)
mask[x,y,:]=1
mask[x1,y1,:]=0
'''
mask22=255*mask
mask22=mask22.astype("uint8")
cv2.imwrite("mask.jpg", mask22)
  '''

target=target.astype("float32")
mask_source=mask_source.astype("float32")
mask=mask.astype("float32")
source=source.astype("float32")

#creat gussian pyramid
kernel = np.array([[1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4],[1,4,6,4,1]])
kernel= kernel/256

#level 0 
target0=target.copy()
mask_source0=mask_source.copy()
mask0=mask.copy()


target0_lap,target0_blured,mask_source0_lap,mask_source0_blured,mask0_blured=make_laplacian(target0,mask_source0,mask0,kernel)
'''
cv2.imwrite("target0_lap.jpg", target0_lap)
cv2.imwrite("target0_blured.jpg", target0_blured)
cv2.imwrite("mask_source0_lap.jpg", mask_source0_lap)
cv2.imwrite("mask_source0_blured.jpg", mask_source0_blured)
cv2.imwrite("mask0_blured.jpg", mask0_blured)'''


#level 1
cols , rows= target0[:,:,0].shape
target1=cv2.pyrDown(target,dstsize= (rows // 2,cols // 2))
mask_source1=cv2.pyrDown(mask_source0, dstsize= (rows // 2,cols // 2))
mask1=cv2.pyrDown(mask0, dstsize= (rows // 2,cols // 2))

target1_lap,target1_blured,mask_source1_lap,mask_source1_blured,mask1_blured=make_laplacian(target1,mask_source1,mask1,kernel)
'''
cv2.imwrite("target1_lap.jpg", target1_lap)
cv2.imwrite("target1_blured.jpg", target1_blured)
cv2.imwrite("mask_source1_lap.jpg", mask_source1_lap)
cv2.imwrite("mask_source1_blured.jpg", mask_source1_blured)
cv2.imwrite("mask1_blured.jpg", mask1_blured)'''


#level 2
cols , rows= target1[:,:,0].shape
target2=cv2.pyrDown(target1, dstsize= (rows // 2,cols // 2))
mask_source2=cv2.pyrDown(mask_source1, dstsize= (rows // 2,cols // 2))
mask2=cv2.pyrDown(mask1, dstsize= (rows // 2,cols // 2))

target2_lap,target2_blured,mask_source2_lap,mask_source2_blured,mask2_blured=make_laplacian(target2,mask_source2,mask2,kernel)
'''
cv2.imwrite("target2_lap.jpg", target2_lap)
cv2.imwrite("target2_blured.jpg", target2_blured)
cv2.imwrite("mask_source2_lap.jpg", mask_source2_lap)
cv2.imwrite("mask_source2_blured.jpg", mask_source2_blured)
cv2.imwrite("mask2_blured.jpg", mask2_blured)'''


#level 3
cols , rows= target2[:,:,0].shape
target3=cv2.pyrDown(target2, dstsize= (rows // 2,cols // 2))
mask_source3=cv2.pyrDown(mask_source2, dstsize= (rows // 2,cols // 2))
mask3=cv2.pyrDown(mask2, dstsize= (rows // 2,cols // 2))

target3_lap,target3_blured,mask_source3_lap,mask_source3_blured,mask3_blured=make_laplacian(target3,mask_source3,mask3,kernel)
'''cv2.imwrite("target3_lap.jpg", target3_lap)
cv2.imwrite("target3_blured.jpg", target3_blured)
cv2.imwrite("mask_source3_lap.jpg", mask_source3_lap)
cv2.imwrite("mask_source3_blured.jpg", mask_source3_blured)
cv2.imwrite("mask3_blured.jpg", mask3_blured)'''


#level 4
cols , rows= target3[:,:,0].shape
target4=cv2.pyrDown(target3, dstsize= (rows // 2,cols // 2))
mask_source4=cv2.pyrDown(mask_source3, dstsize= (rows // 2,cols // 2))
mask4=cv2.pyrDown(mask3,dstsize= (rows // 2,cols // 2))

target4_lap,target4_blured,mask_source4_lap,mask_source4_blured,mask4_blured=make_laplacian(target4,mask_source4,mask4,kernel)
'''
cv2.imwrite("target4_lap.jpg", target4_lap)
cv2.imwrite("target4_blured.jpg", target4_blured)
cv2.imwrite("mask_source4_lap.jpg", mask_source4_lap)
cv2.imwrite("mask_source4_blured.jpg", mask_source4_blured)
cv2.imwrite("mask4_blured.jpg", mask4_blured)'''


#now we can featred them 

dst4_blured=mask4_blured*mask_source4_blured+(1-mask4_blured)*target4_blured
dst4_lap=mask4_blured*mask_source4_lap+(1-mask4_blured)*target4_lap

dst4=dst4_lap+dst4_blured
'''
dst4=dst4.astype("uint8")
cv2.imwrite("dst4.jpg", dst4)
'''
#level3
cols , rows= dst4[:,:,0].shape
dst3=cv2.pyrUp(dst4, dstsize= (rows * 2,cols * 2))
dst3_lap=mask3_blured*mask_source3_lap+(1-mask3_blured)*target3_lap
dst3=dst3+dst3_lap
'''
dst3=dst3.astype("uint8")
cv2.imwrite("dst3.jpg", dst3)
'''
#level2
cols , rows= dst3[:,:,0].shape
dst2=cv2.pyrUp(dst3, dstsize= (rows * 2,cols * 2))
dst2_lap=mask2_blured*mask_source2_lap+(1-mask2_blured)*target2_lap
dst2=dst2+dst2_lap

#level1
cols , rows= dst2[:,:,0].shape
dst1=cv2.pyrUp(dst2, dstsize= (rows * 2,cols * 2))
dst1_lap=mask1_blured*mask_source1_lap+(1-mask1_blured)*target1_lap

dst1=dst1+dst1_lap
#dst1=255*(dst1/dst1.max())
'''
dst1=dst1.astype("uint8")
cv2.imwrite("dst1.jpg", dst1)
'''

#level0
cols , rows= dst1[:,:,0].shape
dst=cv2.pyrUp(dst1, dstsize= (rows * 2,cols * 2))
dst_lap=mask0_blured*mask_source0_lap+(1-mask0_blured)*target0_lap
dst=dst+dst_lap
dst=dst.astype("uint8")
cv2.imwrite("res2.jpg", dst)

