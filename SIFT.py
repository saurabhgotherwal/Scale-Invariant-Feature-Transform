
# coding: utf-8

# In[13]:


import numpy as np
import math
import cv2

INPUT_FOLDER = ''
INPUT_IMAGE_NAME = 'sift.jpg'
OUTPUT_FOLDER = 'output'

def ScaleDownImageByHalf(image):
    scaleDownImage = image[1::2, 1::2]
    return(scaleDownImage)

def GetMatrixSum(matrix):
    sum = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            sum += matrix[i][j]    
    return sum

def GetSigmaMatrix():
    sigmaMatrix = np.asarray([[0.7071,1.0,1.4142,2.0,2.8284],
                            [1.4142,2.0,2.8284,4.0,5.6568],
                            [2.8284,4.0,5.6568,8.0,11.3137],
                            [5.6568,8.0,11.3137,16.0,22.6274]]).reshape(4,5)
    return sigmaMatrix                      
              
def GetKernelValue(x, y, sigma):
    value = (1/(2*(math.pi)*(sigma**2)))*math.exp(-1*(x**2 + y**2)/(2*(sigma**2)))
    return value
             
def GetGaussianKernel(sigma):
    kernelValueList = []
    
    sum = 0
    for column in range(3,-4,-1):
        for row in range(-3,4,1):        
            value = GetKernelValue(row,column,sigma)
            kernelValueList.append(value)
            sum += value
    gaussianKernel = np.asarray(kernelValueList).reshape(7,7)
    
    gaussianKernel /= sum
    
    return gaussianKernel

def GetPadddedImage(image):
    image_h = image.shape[0]
    image_w = image.shape[1]    
    imagePadded = np.zeros((image_h + 6, image_w + 6),dtype = np.uint8)
    imagePadded[3:(imagePadded.shape[0]-3), 3:(imagePadded.shape[1]-3)] = image                                    
            
    return imagePadded

def ApplyGaussianKernel(image,kernel):
    image_padded = GetPadddedImage(image)
        
    image_h = image.shape[0]
    image_w = image.shape[1]
       
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]
    
    h = kernel_h//2
    w = kernel_w//2

    blurImage = np.zeros(image.shape,dtype = np.uint8)
    
    for i in range(h, image_padded.shape[0]-h):
        for j in range(w, image_padded.shape[1]-w):
            sum = 0            
            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum += kernel[m][n]*image_padded[i+m-h][j+n-w]
            blurImage[i-h][j-w] = sum
    
    return blurImage

def GetDog(gaussianOne, gaussianTwo,index,index1):
    dog = np.zeros((gaussianOne.shape[0], gaussianOne.shape[1]),dtype = np.uint8)
    
    for i in range(0, gaussianOne.shape[0]):
        for j in range(0, gaussianOne.shape[1]):
            dog[i][j] = gaussianOne[i][j] - gaussianTwo[i][j]
       
    return dog

def GetOctave(image,sigmas):
    octave = []
    for i in range(0,5):
        kernel = GetGaussianKernel(sigmas[i])
        image = ApplyGaussianKernel(image, kernel)        
        octave.append(image)
    return octave

def GenerateOctaves(image):
    sigmaMatrix = GetSigmaMatrix()
    octaves = []
    for i in range(0,4):
        print('Creating Octave' + str(i+1))
        octave = GetOctave(image,sigmaMatrix[i])
        octaves.append(octave)
        image = ScaleDownImageByHalf(image)
    
    print('Octave creation completed')
    return octaves


def ComputeDog(octave,index):
    dogs = []
    
    for i in range(0,4):
        dog = GetDog(octave[i],octave[i+1],i,index)
        dogs.append(dog)         
    return dogs

def GenerateDog(octaves):    
    dogs = []
    for i in range(0,4):
        print('Computing Dogs for octave' + str(i+1))
        dog = ComputeDog(octaves[i],i)        
        dogs.append(dog) 
    print('DoG creation completed')
    return dogs

def IsMaximaOrMinima(imageAbove,imageSame,imageBelow, pixel):
    result = False
    list = []    
    for i in range(0, imageSame.shape[0]):
        for j in range(0, imageSame.shape[1]):
            list.append(imageAbove[i][j])
            if(i!= 1 and j != 1):
                list.append(imageSame[i][j])
            list.append(imageBelow[i][j])              
    list.sort()
    
    if(list[0] > pixel or list[len(list) - 1] < pixel):
        result = True
        
    return result

def DetectKeyPoints(dogs, image,index):
        
    for dogsIndex in (1,2):  
        for i in range(0, dogs[dogsIndex].shape[0]-2):
            for j in range(0, dogs[dogsIndex].shape[1]-2):
                dogAbove = dogs[dogsIndex-1]
                dogSame = dogs[dogsIndex]
                dogBelow = dogs[dogsIndex+1]
                
                result = IsMaximaOrMinima(dogAbove[i:i+3,j:j+3],
                                          dogSame[i:i+3,j:j+3],
                                          dogBelow[i:i+3,j:j+3],
                                          dogs[dogsIndex][i+1,j+1])
                
                if(result):                                        
                    indexRow = (i+1)*2**index
                    indexColumn = (j+1)*2**index
                    image[indexRow][indexColumn]=255 
                    result = False 
    return image

def DetectPeaks(dogs,image):    
    index = 0
    print('Searching keypoints in image')
    for dog in dogs:
        image = DetectKeyPoints(dog,image,index)
        index = index + 1
    print('Keypoints search completed')
    return image

def writeOutput(octaves, dogs, keypoints):    
    
    print('Writing output in folder: ' + OUTPUT_FOLDER)
    i = 1
    j = 1
    for octave in octaves:
        for image in octave:
            name = OUTPUT_FOLDER + 'Octave_' + str(i) + '_Image_' + str (j) + '.jpg' 
            cv2.imwrite(name,image)
            j += 1
        j = 1
        i += 1
    
    i = 1
    j = 1
                
    for dog in dogs:
        for image in dog:
            name = OUTPUT_FOLDER + 'Octave_' + str(i) + '_Dog_' + str (j) + '.jpg'
            cv2.imwrite(name,image)
            j += 1
        j = 1
        i += 1
    
    cv2.imwrite(OUTPUT_FOLDER + 'Keypoints.jpg',keypoints)
    
    print('Output successfully written in folder: ' + OUTPUT_FOLDER)
    
    return


def SIFT():

    print('Process started')
	
    inputImage = np.array(cv2.imread(INPUT_FOLDER + INPUT_IMAGE_NAME,0))
    
    octaves = GenerateOctaves(inputImage)    
    
    dogs = GenerateDog(octaves)
    
    inputImageCopy = inputImage.copy
    
    keypoints = DetectPeaks(dogs,inputImage)  
    
    writeOutput(octaves,dogs,keypoints)
    
    cv2.imshow('keypoints',keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return

SIFT()


# In[ ]:




