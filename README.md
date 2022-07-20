# image-processing
**develop a program to display the image  using maplot lim**

import  matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread("plants.jpg")
plt.imshow(img)
<br>


**output**
![image](https://user-images.githubusercontent.com/98379636/173807666-3e34e188-5958-4541-b78b-1fd5ab7a0a19.png)

**develop a program to display grayscale image using read and write operation**

import cv2
img=cv2.imread('leaf.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
<br>

**output**
![image](https://user-images.githubusercontent.com/98379636/173809118-3ced8e51-adca-4d45-9492-0d1deb0d2270.png)

**develop a program to perform  linear transformation**
from PIL import Image
img=Image.open("butterfly.jpg")
img=img.rotate(180)
img.show()
cv2.waitkey(0)
cv2.destroyAllWindows()
<br>

**output**
![image](https://user-images.githubusercontent.com/98379636/173812041-90023fe5-c81f-4c11-b209-dbcec7166c9e.png)

**develop a program to convert color string to RGB color values**
from PIL import ImageColor
img1=ImageColor.getrgb("yellow")
print(img1)
img2=ImageColor.getrgb("red")
print(img2)
<br>

**output**
![image](https://user-images.githubusercontent.com/98379636/173816890-df46b402-dba1-4e2c-beb6-e6c2199df74b.png)

**develop program to create image using color**

from PIL import Image
img=Image.new('RGB',(200,400),(255,255,0))
img.show()
<br>

**output**
![image](https://user-images.githubusercontent.com/98379636/173818248-46edcf1e-1469-4ed5-9f6d-75ce0e62dcdd.png)

**develop a program to visualaize the image using various color spaces**
import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('flower.jpg')
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()

img=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
plt.imshow(img)
plt.show()
<br>

**output**
![image](https://user-images.githubusercontent.com/98379636/174038855-7f2f3122-23b2-4d45-8a83-320e2b34c7ce.png)
![image](https://user-images.githubusercontent.com/98379636/174039208-f7eeaf8d-318e-477a-bfe5-1dee2d8bf280.png)
<br>
**develop the image attributes**
from PIL import Image
image=Image.open('leaf.jpg')
print("Filename:",image.filename)
print("Format:",image.format)
print("Mode:",image.mode)
print("size:",image.size)
print("Width:",image.width)
print("Height:",image.height)
image.close()
<br>

**output**
Filename: leaf.jpg
Format: JPEG
Mode: RGB
size: (275, 183)
Width: 275
Height: 183
<br>

**resize the image**
import cv2
img=cv2.imread('leaf.jpg')
print('original image length width',img.shape)
cv2.imshow('original image',img)
imgresize=cv2.resize(img,(150,160))
cv2.imshow('Resized image',imgresize)
print('Resized  image lenght width',imgresize.shape)
cv2.waitKey(0)
<br>

**output**
![image](https://user-images.githubusercontent.com/98379636/174053815-c2f64396-e82c-4451-8af2-0d18280a445b.png)
![image](https://user-images.githubusercontent.com/98379636/174054040-e3d2d88d-53c5-4660-9e7d-e8824e71d7c5.png)
original image length width (183, 275, 3)
Resized  image lenght width (160, 150, 3)
<br>

** convert the original image to grayscale and then binary**
import cv2
img=cv2.imread('butterfly.jpg')
cv2.imshow("RGB",img)
cv2.waitKey(0)
img=cv2.imread('butterfly.jpg',0)
cv2.imshow("Gray",img)
cv2.waitKey(0)
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow("Binary",bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
<br>

**output**
![image](https://user-images.githubusercontent.com/98379636/174061189-81144c5d-05a4-4b38-8ca5-eec901b62d84.png)
![image](https://user-images.githubusercontent.com/98379636/174061294-31734fa9-3444-4c1b-bc89-80efed51ac15.png)
![image](https://user-images.githubusercontent.com/98379636/174061435-1678c746-8240-400b-bfbb-2a294a4e526c.png)

**url code**
from skimage import io
import matplotlib.pyplot as plt
url='https://www.thesprucepets.com/thmb/FOLwbR72UrUpF9sZ45RYKzgO8dg=/3072x2034/filters:fill(auto,1)/yellow-tang-fish-508304367-5c3d2790c9e77c000117dcf2.jpg'
image=io.imread(url)
plt.imshow(image)
plt.show()
<br>

**output**
![image](https://user-images.githubusercontent.com/98379636/175009258-2e38cf41-f147-4dbb-8f17-9258018f33d7.png)
<br>

**masking and bluring**
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
img=cv2.imread('fish2.jpg')
plt.imshow(img)
plt.show()
<br>

**output**
![image](https://user-images.githubusercontent.com/98379636/175265766-58df5657-899f-4b4c-8e54-142a45a47459.png)
<br>


import cv2
hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
light_orange=(1,190,200)
dark_orange=(18,255,255)
mask=cv2.inRange(img,light_orange,dark_orange)
result=cv2.bitwise_and(img,img,mask=mask)
plt.subplot(1,2,1)
plt.imshow(mask,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(result)
plt.show()
**output**
![image](https://user-images.githubusercontent.com/98379636/175266169-6d459996-5420-4de2-80d2-86d283028503.png)

light_white=(0,0,200)
dark_white=(145,60,255)
mask_white=cv2.inRange(hsv_img,light_white,dark_white)
result_white=cv2.bitwise_and(img,img, mask=mask_white)
plt.subplot(1,2,1)
plt.imshow(mask_white,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(result_white)
plt.show()
<br>
**output**
![image](https://user-images.githubusercontent.com/98379636/175266773-4fe3c966-c827-4b93-81b4-2c154b403481.png)
<br>

final_mask=mask + mask_white
final_result=cv2.bitwise_and(img,img,mask=final_mask)
plt.subplot(1,2,1)
plt.imshow(final_mask,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(final_result)
plt.show()
<br>
**output**
![image](https://user-images.githubusercontent.com/98379636/175267190-d8e8d0a9-27d6-4dcb-b845-f1ae199344e6.png)

blur=cv2.GaussianBlur(final_result,(7,7), 0)
plt.imshow(blur)
plt.show()
<br>
**output**
![image](https://user-images.githubusercontent.com/98379636/175268334-62299c3d-23f6-44d5-9d9f-66ee4c149371.png)

**arithmetic operation**
import cv2
import matplotlib.image as mapimg
import matplotlib.pyplot as plt
img1=cv2.imread('image1.jpg')
img2=cv2.imread('image2.jpg')
fimg1=img1+img2
plt.imshow(fimg1)
plt.show()
cv2.imwrite('output.jpg',fimg1)
fimg2=img1-img2
plt.imshow(fimg2)
plt.show()
cv2.imwrite('output.jpg',fimg2)
fimg3=img1*img2
plt.imshow(fimg3)
plt.show()
cv2.imwrite('output.jpg',fimg3)
fimg4=img1/img2
plt.imshow(fimg4)
plt.show()
cv2.imwrite('output.jpg',fimg4)

**output**
![image](https://user-images.githubusercontent.com/98379636/175273850-5c7dad34-955f-4405-9c57-56e09e5a7b6d.png)
![image](https://user-images.githubusercontent.com/98379636/175274176-f118e509-006f-474f-87cf-cedf27b51698.png)
![image](https://user-images.githubusercontent.com/98379636/175274280-285c650c-51f5-4fd8-84f1-d994a3f78761.png)

**2D array**
import cv2 as c
import numpy as np
from PIL import Image
array =np.zeros([100,200,3],dtype=np.uint8)
array[:,:100]=[255,130,0]
array[:,100:]=[0,0,255]
img=Image.fromarray(array)
img.save('img1.jpg')
img.show()
c.waitKey(0)

**output**
![image](https://user-images.githubusercontent.com/98379636/175282663-d4283469-a74a-4387-b5ea-6d3f72eb18b9.png)

**change image to different color spaces**
import cv2
img=cv2.imread("plants.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
cv2.imshow("GRAY image",gray)
cv2.imshow("HSV image",hsv)
cv2.imshow("LAB image",lab)
cv2.imshow("HLS image",hls)
cv2.imshow("YUV image",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()

**output**
![image](https://user-images.githubusercontent.com/98379636/175287156-63845181-57fd-4aff-90db-a14ecd54953f.png)
![image](https://user-images.githubusercontent.com/98379636/175287272-fde770fc-2b54-4174-b12c-494d284430f8.png)
![image](https://user-images.githubusercontent.com/98379636/175287355-53233039-b1da-4198-b7bc-acbd4c0c8ead.png)
![image](https://user-images.githubusercontent.com/98379636/175287467-13195a4d-fb76-47f9-90f2-4636edeb2f49.png)
![image](https://user-images.githubusercontent.com/98379636/175287573-027de779-eb23-4a1c-b586-829ef1061d6b.png)

bluring
import cv2
import numpy as np
image=cv2.imread('dog.jpg')
cv2.imshow('Original Image',image)
cv2.waitKey(0)
#Gaussian Blur
Gaussian=cv2.GaussianBlur(image,(7,7),0)
cv2.imshow('Gaussian Blurring',Gaussian)
#Median BLUR
median=cv2.medianBlur(image,5)
cv2.imshow('Median Blurring',median)
cv2.waitKey(0)
#Bilateral Blur
bilateral=cv2.bilateralFilter(image,9,75,75)
cv2.imshow('Bilateral Blurring',bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()

output

![image](https://user-images.githubusercontent.com/98379636/179921789-c03dbca7-9da4-4b58-8cd4-3673e4c46711.png)
![image](https://user-images.githubusercontent.com/98379636/179921913-2411eef6-086a-4b47-9252-b5586e2513f0.png)
![image](https://user-images.githubusercontent.com/98379636/179922176-f6385c88-3fb6-47c4-be93-e27c12696ab6.png)
![image](https://user-images.githubusercontent.com/98379636/179922320-9bac4ebe-a893-473e-9ffe-1eb1ea6e7121.png)
![image](https://user-images.githubusercontent.com/98379636/179922461-3bdc44c9-268e-4a1a-a6c5-afa0627ba35e.png)
![image](https://user-images.githubusercontent.com/98379636/179924089-5c835363-cbc1-4730-bea1-d4f5e0c72bec.png)


image negative
#matplotlib inline
import imageio
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
pic=imageio.imread('flowers.jpg')
plt.figure(figsize=(6,6))
plt.imshow(pic);
plt.axis('off');

output
![image](https://user-images.githubusercontent.com/98379636/179958046-959b43c0-ec18-4e40-bee4-df62f9725fc1.png)

negative=255-pic # neg=(L-1) - img
plt.figure(figsize=(6,6))
plt.imshow(negative);
plt.axis("off");

output
![image](https://user-images.githubusercontent.com/98379636/179959116-fd8d7f47-ba1d-41fe-9af9-b79a30ed04ea.png)

#matplotlib inline
import imageio
import numpy as np
import matplotlib.pyplot as plt
pic=imageio.imread('butterfly.jpg')
gray=lambda rgb :np.dot(rgb[...,:3],[0.299,0.587,0.114])
gray=gray(pic)
max_=np.max(gray)
def log_transform():
    return(255/np.log(1+max_))*np.log(1+gray)
plt.figure(figsize=(5,5))
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))
plt.axis('off');

output
![image](https://user-images.githubusercontent.com/98379636/179959721-14091578-d287-43d2-875c-8ea3b790ad17.png)

import imageio
import matplotlib.pyplot as plt
# gamma encoding
pic=imageio.imread('butterfly.jpg')
gamma=2.2# Gamma < 1 Dark; Gamma > 1 bright
gamma_correction=((pic/255)**(1/gamma))
plt.figure(figsize=(5,5))
plt.imshow(gamma_correction)
plt.axis('off');

output
![image](https://user-images.githubusercontent.com/98379636/179961155-54e29d67-649c-4bc8-b315-8d8526e3546a.png)

program for image manipulation
#image sharpen
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
#load the image
my_image=Image.open('images.jpg')
#use sharpen function
sharp=my_image.filter(ImageFilter.SHARPEN)
#save the image
sharp.save('E:\dog.jpg')
sharp.show()
plt.imshow(sharp)
plt.show()

output
![image](https://user-images.githubusercontent.com/98379636/179965821-9ddb4329-0469-4250-8148-04227c992d3a.png)

flipping
#image flip
import matplotlib.pyplot as plt
#load the image
img=Image.open('images.jpg')
plt.imshow(img)
plt.show()
#use the flip function
flip=img.transpose(Image.FLIP_LEFT_RIGHT)
#save the image
flip.save('E:\image_flip.jpg')
plt.imshow(flip)
plt.show()
output
![image](https://user-images.githubusercontent.com/98379636/179966032-ee3d65e3-71c0-4d89-bac5-ef3a0ff551ac.png)






















































