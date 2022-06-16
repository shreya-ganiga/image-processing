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

**resize**
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















