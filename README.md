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
































