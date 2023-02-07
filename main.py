import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from Affiliated import denoise, deconv, motion_blur, addNoise
from Affiliated import utils, auxiliary
from skimage.metrics import mean_squared_error
from Kernel_Estimation import kernel_estimation
from skimage.util import random_noise
from scipy import signal
#from medpy.filter.smoothing import anisotropic_diffusion
import cv2
from csv import writer
 

#import pandas as pd

#file_path = glob.glob('/home/venkat/Downloads/Image-deblur-using-image-pairs-master/Data Set') # Read out all the test image in the /image/ file directory
img_dir = '/home/venkat/Data_Set/' 
data_path = os.path.join(img_dir,"./*tif") 
files = glob.glob(data_path) 
num_to_cal = 3 # This parameter is used to control the num of the image you want to read out from the ./images datafile
num = 0
files = files[2:num_to_cal] 
is_random_kernel = True    # Decide wheather to generate the random kernel
#size_of_kernel = 3 
#niter=20
#kappa=1
#gamma=0.25++
#voxelspacing=None
                    
#option=1

# Decide the size of the kernel generated and estimate


for i,fname in enumerate(files):
    
    
    # Read the images form the folder in the given path
    savedir = '/home/venkat/Downloads/Image-deblur-using-image-pairs-master' + str(19000+num)+'/'
    csvname = savedir + "IQE_%s.csv" % (os.path.basename(fname))   # Used to estimate the quality of the image, Image quality estimation
    # create a directory to store the results if it does not exists 
    if not os.path.exists(savedir):
        os.mkdir(savedir)    
        
    f= open(csvname,mode = 'w') 
        
    f_csv = csv.writer(f, delimiter = ',')
        
    f_csv.writerow(['i','Method','size_of_kernel','lambda','SSIM','PSNR', 'MSE'])
    
    # Initialize different kernel sizes 
            
    for size_of_kernel in np.arange(5,7,2):   
        
        #f_csv.writerow(['='])
        
               
    # initialize regularization parameter  
    
        for la in np.linspace(0.1,0.2,1):    
            
            #f_csv.writerow(['='])
    
    
            print("--"*10)
            print("Starting the {} round".format(num+1))
        #    savename = 'eval_' + str(2000+num)
            
            
            I = io.imread(fname,as_gray=True)
            I = img_as_float(I)
            
       # Generate Random kernel (True) to blur the image 
       
            if is_random_kernel:
                blur_kernel = utils.kernel_generator(size_of_kernel)
               # B = cv2.blur(I, (np.int16(blur_kernel), np.int16(blur_kernel))
                #B=cv2.GaussianBlur(I, (size_of_kernel, size_of_kernel), 0)
                B=cv2.filter2D(I, -1, blur_kernel)
                #B=cv2.blur(I, (size_of_kernel, size_of_kernel))
            else:
                motion_degree = np.random.randint(0,360)    # Generate one specific motion deblur
                B , blur_kernel = motion_blur(I,size_of_kernel,motion_degree)
      
        # Add noise to the image      
        
            N=addNoise(I,0,0.01)    
            #N = I+ random_noise(I, mode='s&p', amount=0.1)
            Nd = denoise(N)
            #Nd=cv2.Laplacian(B,cv2.CV_64F)
           #Nd=anisotropic_diffusion(N, niter, kappa, gamma, voxelspacing, option)
      
        # Estimating the Kerne  
        
            K_estimated = kernel_estimation(Nd,B,lens=size_of_kernel,lam=la,method='l1ls')
            #K_estimated1 = blur_kernel
            
            auxiliary.kernel_write(np.float32(K_estimated),"estimated (img=%d,size_of_kernel=%d, lambda=%d).tiff" % (i,size_of_kernel,la),savedir)
            auxiliary.kernel_write(blur_kernel,"true (img=%d, size_of_kernel=%d,lambda=%d).tiff" % (i,size_of_kernel,la),savedir)
       
        # Save the results
        
            plt.imsave(savedir+"original(img=%d,size_of_kernel=%d, lambda=%d).tiff" % (i,size_of_kernel,la),I,cmap = 'gray')
            plt.imsave(savedir+"blurred(img=%d,size_of_kernel=%d, lambda=%d).tiff" % (i,size_of_kernel,la),B,cmap = 'gray')
            plt.imsave(savedir+"denoised(img=%d, size_of_kernel=%d,lambda=%d).tiff" % (i,size_of_kernel,la),Nd,cmap = 'gray')
        
        
            
    
            
            deconvmode = ['detailedRL','lucy','resRL','gcRL']
        ## deconvolution, can be 'lucy', 'resRL', 'gcRL' and 'detailedRL'.
            
            for demode in deconvmode:
                # deBlur = deblur(Nd,B,unikernel = True,deconvmode=demode)
                deBlur = deconv(Nd,B,K_estimated,mode=demode)
                
                plt.imsave(savedir+"deblurred_" +demode+'(img=%d, size_of_kernel=%d, lambda=%d)' %(i,size_of_kernel,la) +".tiff",deBlur,cmap = 'gray')
                
                ssim1 = ssim(I,deBlur)
                psnr1 = peak_signal_noise_ratio(I,deBlur)
                mse1=mean_squared_error(I, deBlur)
               
                result = [i,demode,size_of_kernel,la,ssim1,psnr1, mse1]  
               #result=result.insert(1, demode)
                f= open(csvname,mode = 'a+')
                f_csv = csv.writer(f,delimiter = ',')
                f_csv.writerow(result)
        
            
# =============================================================================
#                find percentage of match between estimated kernel pixel value
#                to True kernel pixel value 
                 
# =============================================================================
        
        h,w =blur_kernel.shape[:2]
        total=0
        sum=h*w
        
        for i in range(h):
             for j in range(w):
                 if K_estimated[i,j]-blur_kernel[i,j]<0.001:
                     total+=1
        
        matching_percentage=(total/sum)*100
                #print("percenatge of matching pixels={}".format(matching_percenatge))
        
    
        
        csv_name1=savedir + "%s.csv" % (os.path.basename(fname))
        
        f1=open(csv_name1, 'w')
        
        f1_csv= csv.writer(f1)
            
        f1_csv.writerow(['Percenatge of matching pixels'])
        
        f1= open(csv_name1, mode = 'a+')
        
        f1_csv = csv.writer(f1)
        
        f1_csv.writerow([matching_percentage])
        
        
                
    num += 1        
    print("Complete the {} round".format(num))
    print("__"*10)
    print("Complete all cycles")
    print("__"*10)
    
    
    
#cor = signal.correlate2d (np.float32(K_estimated), blur_kernel)
#print(cor)  
    
               
        

