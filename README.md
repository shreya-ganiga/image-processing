# image-processing
**develop a program to display the image  using maplot lim**

import  matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread("plants.jpg")
plt.imshow(img)


**output**
![image](https://user-images.githubusercontent.com/98379636/173807666-3e34e188-5958-4541-b78b-1fd5ab7a0a19.png)

**develop a program to display grayscale image using read and write operation**

import cv2
img=cv2.imread('leaf.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

**output**
![image](https://user-images.githubusercontent.com/98379636/173809118-3ced8e51-adca-4d45-9492-0d1deb0d2270.png)

**develop a program to perform  linear transformation**
from PIL import Image
img=Image.open("butterfly.jpg")
img=img.rotate(180)
img.show()
cv2.waitkey(0)
cv2.destroyAllWindows()

**output**



