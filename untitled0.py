#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 10:59:00 2022

@author: venkat
"""
import os
import csv
import cv2
import numpy as np
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from Affiliated import denoise, deconv, motion_blur, addNoise
from Affiliated import utils, auxiliary
from skimage.metrics import mean_squared_error
from Kernel_Estimation import kernel_estimation
import matplotlib.pyplot as plt
num=1     
path=cv2.imread('/home/venkat/Data_Set/Img1.tif')
savedir = '/home/venkat/Downloads/Image-deblur-using-image-pairs-master' + str(3000+num)+'/'

csvname = savedir + "IQE_%s.csv" % (os.path.basename(path))   # Used to estimate the quality of the image, Image quality estimation
  
f= open(csvname,mode = 'w') 
    
f_csv = csv.writer(f, delimiter = ',')
    
f_csv.writerow(['i','Method','size_of_kernel','lambda','SSIM','PSNR', 'MSE'])           
    
 
 
print("--"*10)
print("Starting the {} round".format(num+1))
     #    savename = 'eval_' + str(2000+num)
la=1         
I = cv2.imread(csvname,as_gray=True)
I =img_as_float(I)
is_random_kernel=True
size_of_kernel=7
if is_random_kernel:
             #blur_kernel = utils.kernel_generator(size_of_kernel)
             #B = utils.blur(I,blur_kernel)
    B=cv2.GaussianBlur(I, (size_of_kernel,size_of_kernel),0)
             #B=cv2.blur(I, (size_of_kernel,size_of_kernel))
else:
    motion_degree = np.random.randint(0,360)    # Generate one specific motion deblur
    B , blur_kernel = motion_blur(I,size_of_kernel,motion_degree)
             
    N = addNoise(I,0,0.01)
    Nd = denoise(N)
         #Nd=cv2.Laplacian(B,cv2.CV_64F)
        #Nd=anisotropic_diffusion(N, niter, kappa, gamma, voxelspacing, option)
         
K_estimated = kernel_estimation(Nd,B,lens=size_of_kernel,lam=1,method='l1ls')
         #K_estimated = blur_kernel
         
auxiliary.kernel_write(K_estimated,"estimated (size_of_kernel=%d, lambda=%d)" % (size_of_kernel,1),savedir)
         #auxiliary.kernel_write(blur_kernel,"true (img=%d, size_of_kernel=%d,lambda=%d)" % (i,size_of_kernel,la),savedir)
     
plt.imsave(savedir+"original(size_of_kernel=%d, lambda=%d).png" % (size_of_kernel,la),I,cmap = 'gray')
plt.imsave(savedir+"blurred(size_of_kernel=%d, lambda=%d).png" % (size_of_kernel,la),B,cmap = 'gray')
plt.imsave(savedir+"denoised(size_of_kernel=%d,lambda=%d).png" % (size_of_kernel,la),Nd,cmap = 'gray')
    
     
         
 
         
deconvmode = ['detailedRL','lucy','resRL','gcRL']
     ## deconvolution, can be 'lucy', 'resRL', 'gcRL' and 'detailedRL'.
         
for demode in deconvmode:
             # deBlur = deblur(Nd,B,unikernel = True,deconvmode=demode)
   deBlur = deconv(Nd,B,K_estimated,mode=demode)
             
   plt.imsave(savedir+"deblurred_" +demode+'(size_of_kernel=%d, lambda=%d)' %(size_of_kernel,la) +".tif",deBlur,cmap = 'gray')
             
   ssim1 = ssim(I,deBlur)
   psnr1 = peak_signal_noise_ratio(I,deBlur)
   mse1=mean_squared_error(I, deBlur)
            
   result = [demode,size_of_kernel,la,ssim1,psnr1, mse1]  
            #result=result.insert(1, demode)
   f= open(csvname,mode = 'a+')
   f_csv = csv.writer(f,delimiter = ',')
   f_csv.writerow(result)
     
 
             
             
num += 1        
print("Complete the {} round".format(num))
print("__"*10)
print("Complete all cycles")
print("__"*10)