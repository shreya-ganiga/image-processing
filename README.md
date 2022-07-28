# image-processing
**develop a program to display the image  using maplot lim**<br>
import  matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread("plants.jpg")<br>
plt.imshow(img)<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/98379636/173807666-3e34e188-5958-4541-b78b-1fd5ab7a0a19.png)<br>

**develop a program to display grayscale image using read and write operation**<br>
import cv2<br>
img=cv2.imread('leaf.jpg',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
<br>
**output**<br>
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
![image](https://user-images.githubusercontent.com/98379636/173812041-90023fe5-c81f-4c11-b209-dbcec7166c9e.png)<br>

**develop a program to convert color string to RGB color values**<br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/98379636/173816890-df46b402-dba1-4e2c-beb6-e6c2199df74b.png)<br>
**develop program to create image using color**<br>
from PIL import Image<br>
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/98379636/173818248-46edcf1e-1469-4ed5-9f6d-75ce0e62dcdd.png)<br>

**develop a program to visualaize the image using various color spaces**<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('flower.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
plt.imshow(img)<br>
plt.show()<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/98379636/174038855-7f2f3122-23b2-4d45-8a83-320e2b34c7ce.png)<br>
![image](https://user-images.githubusercontent.com/98379636/174039208-f7eeaf8d-318e-477a-bfe5-1dee2d8bf280.png)<br>
<br>
**develop the image attributes**<br>
from PIL import Image<br>
image=Image.open('leaf.jpg')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("size:",image.size)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
image.close()<br>
<br>
**output**<br>
Filename: leaf.jpg<br>
Format: JPEG<br>
Mode: RGB<br>
size: (275, 183)<br>
Width: 275<br>
Height: 183<br>
<br>
**resize the original image**<br>
import cv2<br>
img=cv2.imread('leaf.jpg')<br>
print('original image length width',img.shape)<br>
cv2.imshow('original image',img)<br>
imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('Resized image',imgresize)<br>
print('Resized  image lenght width',imgresize.shape)<br>
cv2.waitKey(0)<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/98379636/174053815-c2f64396-e82c-4451-8af2-0d18280a445b.png)<br>
![image](https://user-images.githubusercontent.com/98379636/174054040-e3d2d88d-53c5-4660-9e7d-e8824e71d7c5.png)<br>
original image length width (183, 275, 3)<br>
Resized  image lenght width (160, 150, 3)<br>
<br>
** convert the original image to grayscale and then binary**<br>
import cv2<br>
img=cv2.imread('butterfly.jpg')<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>
img=cv2.imread('butterfly.jpg',0)<br>
cv2.imshow("Gray",img)<br>
cv2.waitKey(0)<br>
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
<br>
**output**
![image](https://user-images.githubusercontent.com/98379636/174061189-81144c5d-05a4-4b38-8ca5-eec901b62d84.png)<br>
![image](https://user-images.githubusercontent.com/98379636/174061294-31734fa9-3444-4c1b-bc89-80efed51ac15.png)<br>
![image](https://user-images.githubusercontent.com/98379636/174061435-1678c746-8240-400b-bfbb-2a294a4e526c.png)<br>

**url code**
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://www.thesprucepets.com/thmb/FOLwbR72UrUpF9sZ45RYKzgO8dg=/3072x2034/filters:fill(auto,1)/yellow-tang-fish-508304367-5c3d2790c9e77c000117dcf2.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/98379636/175009258-2e38cf41-f147-4dbb-8f17-9258018f33d7.png)<br>
<br>
* write a program to masking and bluring**
import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('fish2.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
<br>

**output**<br>
![image](https://user-images.githubusercontent.com/98379636/175265766-58df5657-899f-4b4c-8e54-142a45a47459.png)<br>
<br>

import cv2<br>
hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_orange=(1,190,200)<br>
dark_orange=(18,255,255)<br>
mask=cv2.inRange(img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>
**output**<br>
![image](https://user-images.githubusercontent.com/98379636/175266169-6d459996-5420-4de2-80d2-86d283028503.png)<br>

light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img, mask=mask_white)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>
<br>
**output**
![image](https://user-images.githubusercontent.com/98379636/175266773-4fe3c966-c827-4b93-81b4-2c154b403481.png)<br>
<br>

final_mask=mask + mask_white<br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(final_mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(final_result)<br>
plt.show()<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/98379636/175267190-d8e8d0a9-27d6-4dcb-b845-f1ae199344e6.png)<br>

blur=cv2.GaussianBlur(final_result,(7,7), 0)<br>
plt.imshow(blur)<br>
plt.show()<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/98379636/175268334-62299c3d-23f6-44d5-9d9f-66ee4c149371.png)<br>

** write a progam to perform arithmetic operation on edges**<br>
import cv2<br>
import matplotlib.image as mapimg<br>
import matplotlib.pyplot as plt<br>
img1=cv2.imread('image1.jpg')<br>
img2=cv2.imread('image2.jpg')<br>
fimg1=img1+img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg1)<br>
fimg2=img1-img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg2)<br>
fimg3=img1*img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg3)<br>
fimg4=img1/img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg4)<br>
**output**<br>
![image](https://user-images.githubusercontent.com/98379636/175273850-5c7dad34-955f-4405-9c57-56e09e5a7b6d.png)<br>
![image](https://user-images.githubusercontent.com/98379636/175274176-f118e509-006f-474f-87cf-cedf27b51698.png)<br>
![image](https://user-images.githubusercontent.com/98379636/175274280-285c650c-51f5-4fd8-84f1-d994a3f78761.png)<br>

** program to create an image using 2D array**<br>
import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array =np.zeros([100,200,3],dtype=np.uint8)<br>
array[:,:100]=[255,130,0]<br>
array[:,100:]=[0,0,255]<br>
img=Image.fromarray(array)<br>
img.save('img1.jpg')<br>
img.show()<br>
c.waitKey(0)<br>
**output**<br>
![image](https://user-images.githubusercontent.com/98379636/175282663-d4283469-a74a-4387-b5ea-6d3f72eb18b9.png)<br>
<br>
**create image to different color spaces**<br>
import cv2<br>
img=cv2.imread("plants.jpg")<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
**output**<br>
![image](https://user-images.githubusercontent.com/98379636/175287156-63845181-57fd-4aff-90db-a14ecd54953f.png)<br>
![image](https://user-images.githubusercontent.com/98379636/175287272-fde770fc-2b54-4174-b12c-494d284430f8.png)<br>
![image](https://user-images.githubusercontent.com/98379636/175287355-53233039-b1da-4198-b7bc-acbd4c0c8ead.png)<br>
![image](https://user-images.githubusercontent.com/98379636/175287467-13195a4d-fb76-47f9-90f2-4636edeb2f49.png)<br>
![image](https://user-images.githubusercontent.com/98379636/175287573-027de779-eb23-4a1c-b586-829ef1061d6b.png)<br>
<br>
**bluring**<br>
import cv2<br>
import numpy as np<br>
image=cv2.imread('dog.<jpg')<br>
cv2.imshow('Original Image',image)<br>
cv2.waitKey(0)<br>
#Gaussian Blur<br>
Gaussian=cv2.GaussianBlur(image,(7,7),0)<br>
cv2.imshow('Gaussian Blurring',Gaussian)<br>
#Median BLUR<br>
median=cv2.medianBlur(image,5)<br>
cv2.imshow('Median Blurring',median)<br>
cv2.waitKey(0)<br>
#Bilateral Blur<br>
bilateral=cv2.bilateralFilter(image,9,75,75)<br>
cv2.imshow('Bilateral Blurring',bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
output<br>
![image](https://user-images.githubusercontent.com/98379636/180190404-80494c1d-d856-4aca-92aa-23e9659fd44c.png)<br>
![image](https://user-images.githubusercontent.com/98379636/180190674-7cc01e40-cadf-46b2-8768-03fd24852beb.png)<br>
![image](https://user-images.githubusercontent.com/98379636/180190810-2587e100-3789-45f9-9031-dae26f8458c0.png)<br>
![image](https://user-images.githubusercontent.com/98379636/180191026-27d96270-0d59-4b02-9875-68ea23ebf505.png)<br>
<br>

**program to perform basic data analysis using intensity transformation
a)image negative
#matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<r>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('flowers.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>
<br>
output<br
![image](https://user-images.githubusercontent.com/98379636/179958046-959b43c0-ec18-4e40-bee4-df62f9725fc1.png)<br>
    <br>
negative=255-pic # neg=(L-1) - img<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(negative);<br>
plt.axis("off");<br>
<br>
output<br>
![image](https://user-images.githubusercontent.com/98379636/179959116-fd8d7f47-ba1d-41fe-9af9-b79a30ed04ea.png)<br>

#matplotlib inline<br>
import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
pic=imageio.imread('butterfly.jpg')<br>
gray=lambda rgb :np.dot(rgb[...,:3],[0.299,0.587,0.114])<br>
gray=gray(pic)<br>
max_=np.max(gray)<br>
def log_transform():<br>
    return(255/np.log(1+max_))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>
output<br>
![image](https://user-images.githubusercontent.com/98379636/179959721-14091578-d287-43d2-875c-8ea3b790ad17.png)<br>

import imageio<br>
import matplotlib.pyplot as plt<br>
# gamma encoding<br>
pic=imageio.imread('butterfly.jpg')<br>
gamma=2.2# Gamma < 1 Dark; Gamma > 1 bright<br>
gamma_correction=((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>
<br>
output<br>
![image](https://user-images.githubusercontent.com/98379636/179961155-54e29d67-649c-4bc8-b315-8d8526e3546a.png)<br>

**program to perform basic image manipulation** <br>
a)sharpen<br>
#image sharpen<br>
from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>
#load the image<br>
my_image=Image.open('images.jpg')<br>
#use sharpen function<br>
sharp=my_image.filter(ImageFilter.SHARPEN)<br>
#save the image<br>
sharp.save('E:\dog.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>
<br>
output<br>
![image](https://user-images.githubusercontent.com/98379636/179965821-9ddb4329-0469-4250-8148-04227c992d3a.png)<br>

flipping<br>
#image flip<br>
import matplotlib.pyplot as plt<br>
#load the image<br>
img=Image.open('images.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
#use the flip function<br>
flip=img.transpose(Image.FLIP_LEFT_RIGHT)<br>
#save the image<br>
flip.save('E:\image_flip.jpg')<br>
plt.imshow(flip)<br>
plt.show()<br>
output<br>
![image](https://user-images.githubusercontent.com/98379636/179966032-ee3d65e3-71c0-4d89-bac5-ef3a0ff551ac.png)<br>
<br>
**cropping**
#importing image class from PIL module<br>
from PIL import Image<br>
import matplotlib.pyplot as plt<br>
#opens a image in RGB mode<br>
im=Image.open('images.jpg')<br>
<br>
#size of the image in pixels(size of original image)<br>
#(this is not mandatory)<br>
width,height=im.size<br>
#cropped image of dimension<br>
#(it will not change orginal image)<br>
<br>
im1=im.crop((10,20,55,70))<br>
#shows the image in image viewer<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>
output<br>
![image](https://user-images.githubusercontent.com/98379636/179968768-23ad53a8-a709-4167-89e9-bd6e075ba333.png)<br>
<br>
image enhancement<br>
from PIL import Image<br>
from PIL import ImageEnhance<br>
image=Image.open('flower.jpg')<br>
image.show()<br>
enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>
image_brightened=enh_bri.enhance(brightness)<br>
image_brightened.show()<br>
enh_col=ImageEnhance.Color(image)<br>
color=1.5<br>
image_colored=enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con=ImageEnhance.Contrast(image)<br>
contrast=1.5<br>
image_contrasted=enh_con.enhance(contrast)<br>
image_contrasted.show()<br>
enh_sha=ImageEnhance.Sharpness(image)<br>
sharpness=3.0<br>
image_sharped=enh_sha.enhance(sharpness)<br>
image_sharped.show()<br>
<br>
output<br>
![image](https://user-images.githubusercontent.com/98379636/179970283-d70b0d29-9395-408b-b673-3c6116a03d93.png)<br>
![image](https://user-images.githubusercontent.com/98379636/179970452-51404fc4-df7b-4949-922c-eded7e7b200e.png)<br>
![image](https://user-images.githubusercontent.com/98379636/179970524-6318c96f-01fc-498d-88da-8aa67f6af139.png)<br>
![image](https://user-images.githubusercontent.com/98379636/179970596-127fcd1c-0919-4bf0-82f9-d5f14536b1bc.png)<br>
![image](https://user-images.githubusercontent.com/98379636/179970684-79609823-b881-499c-8285-3e8ca2760968.png)<br>
<br>
bitwise operation<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('baby.jpg',1)<br>
image2=cv2.imread('baby.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and(image1,image2)<br>
bitwiseOr=cv2.bitwise_or(image1,image2)<br>
bitwiseXor=cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not(image1)<br>
bitwiseNot_img2=cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>
output<br>
![image](https://user-images.githubusercontent.com/98379636/180172665-dacdf86c-c3e9-4b17-ad11-ee2df80f5b21.png)<br>
<br>
morphological operation<br>
import cv2<br>
import numpy a<br>
from matplotlib import pyplot as plt<br>
from PIL import Image,ImageEnhance<br>
img=cv2.imread('images.jpg',0)<br>
ax=plt.subplots(figsize=(20,10))<br>
kernel=np.ones((5,5),np.uint8)<br>
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)<br>
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)<br>
erosion=cv2.erode(img,kernel,iterations = 1)<br>
dilation=cv2.dilate(img,kernel,iterations=1)<br>
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>
<br>
output<br>
![image](https://user-images.githubusercontent.com/98379636/180173112-77d8799c-5ece-4cb0-b106-c8d539889775.png)<br>
<br>
graylevel slicing with background<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('flower.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>
<br>
output<br>
![image](https://user-images.githubusercontent.com/98379636/180176458-7f8853ca-4851-4e87-bbe9-8391dd81f5d8.png)<br>
<br>
graylevel slicing  without background
<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('butterfly.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x): <br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br><br>
    else:<br>
        z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing without background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>
<br>
output<br>
![image](https://user-images.githubusercontent.com/98379636/180176637-ec79a8dc-8ab2-4bd9-9054-d8f08370a5eb.png)<br>
    
  **develop a program to  read the image b)write the grayscale and
    c)display the original image and grayscale image **<br>
    import cv2<br>
OriginalImg=cv2.imread('flowers.jpg')<br>
GrayImg=cv2.imread('flowers.jpg',0)<br><br>
isSaved=cv2.imwrite('E:/i.jpg',GrayImg)<br>
cv2.imshow('Display Original Image',OriginalImg)<br>
cv2.imshow('Display Grayscale Image',GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
        print('The Image is successfully saved')<br>
    output
    ![image](https://user-images.githubusercontent.com/98379636/180194887-255776f6-df5d-433d-83a0-e9453c8c2975.png)<br>
    ![image](https://user-images.githubusercontent.com/98379636/180195003-ed31b11c-d6bc-42e0-b7f8-c67105156e2a.png)<br>
The Image is successfully saved<br>
    
**standard deviation**<br>
from PIL import Image, ImageStat<br>
im = Image.open('rose.jpg')<br>
stat = ImageStat.Stat(im)<br>
print(stat.stddev)<br>
output<br>
[82.54738956211672, 75.14044878977742, 67.40861160301475]<br>
**maximum**<br>
import cv2<br>
import numpy as np<br>
img=cv2.imread('horse1.jpg')<br>
cv2.imshow('horse1.jpg',img)<br>
cv2.waitKey(0)<br>
#max_channels=np.amax([np.amax(img[:,:,0]),np.amax(img[:,:,1]),np.amax(img[:,:,2])])<br>
#print(max_channels)<br>
np.max(img)<br>
output<br>
255<br>
![image](https://user-images.githubusercontent.com/98379636/181211670-fb5dc9fa-1498-42ee-bc0a-5a9f7311658b.png)<br>
**minimum**<br>
import cv2<br>
import numpy as np<br>
img=cv2.imread('rose.jpg')<br>
cv2.imshow('rose.jpg',img)<br>
cv2.waitKey(0)<br>
#min_channels=np.amin([np.amin(img[:,:,0]),np.amin(img[:,:,1]),np.amin(img[:,:,2])])<br>
#print(min_channels)<br>
np.min(img)<br>
0<br>
output<br>
![image](https://user-images.githubusercontent.com/98379636/181211976-688a4978-d143-48d0-aace-3f0a498507b9.png)<br>
**average**<br>
import cv2<br>
import numpy as np<br>
img=cv2.imread('butterfly.jpg')<br>
cv2.imshow('butterfly.jpg',img)<br>
cv2.waitKey(0)<br>
np.average(img)<br>
output<br>
![image](https://user-images.githubusercontent.com/98379636/181212682-39d86ebc-8def-44d7-be70-3be65e40c84b.png)<br>
89.28693453273364<br>
**standard deviation**<br>
import cv2<br>
import numpy as np<br>
img=cv2.imread('cat.jpg')<br>
cv2.imshow('cat.jpg',img)<br>
cv2.waitKey(0)<br>
np.std(img)<br>
output<br>
![image](https://user-images.githubusercontent.com/98379636/181213373-92f3aa3f-4b41-4108-b5a1-b0f0d6cb6365.png)<br>
36.555271183871334<br>

from PIL import Image<br>
from numpy import asarray<br>
img = Image.open('f1.jpg')<br>
numpydata = asarray(img)<br>
print(numpydata)<br>
**output**<br>
[[[254   0   0]
  [254   0   0]
  [254   0   0]
  ...
  [  0   0   0]
  [  0   0   0]
  [  0   0   0]]

 [[254   0   0]
  [254   0   0]
  [254   0   0]
  ...
  [  0   0   0]
  [  0   0   0]
  [  0   0   0]]

 [[254   0   0]
  [254   0   0]
  [254   0   0]
  ...
  [  0   0   0]
  [  0   0   0]
  [  0   0   0]]

 ...

 [[  0   0   0]
  [  0   0   0]
  [  0   0   0]
  ...
  [ 34   0   2]
  [ 25   0   2]
  [ 13   0   4]]

 [[  0   0   0]
  [  0   0   0]
  [  0   0   0]
  ...
  [ 25   0   2]
  [ 20   0   4]
  [  6   0   5]]

 [[  0   0   0]
  [  0   0   0]
  [  0   0   0]
  ...
  [ 13   0   4]
  [  7   0   5]
  [  0   3   5]]]
  
from PIL import Image<br>
import matplotlib.pyplot as plt<br>
input_image = Image.new(mode="RGB", size=(1000, 1000),color="pink")<br>
pixel_map = input_image.load()<br>
width, height = input_image.size<br>
z = 100<br>
for i in range(width):<br><br>
    for j in range(height):<br>
        if((i >= z and i <= width-z) and (j >= z and j <= height-z)):<br>
            pixel_map[i, j] = (230,230,250)<br>
        else:<br>
            pixel_map[i, j] = (216,191,216)<br>
    for i in range(width):<br>
        pixel_map[i, i] = (0, 0, 255)<br>
        pixel_map[i, width-i-1] = (0, 0, 255)<br>
plt.imshow(input_image)<br>
plt.show()<br>
**output**<br>
![image](https://user-images.githubusercontent.com/98379636/181234892-66bcc9ed-4e85-47a3-95a2-9e7414807d5d.png)<br>
                                                                    
import numpy as np<br>
import matplotlib.pyplot as plt<br>
arr = np.zeros((256,256,3), dtype=np.uint8)<br>
imgsize = arr.shape[:2]<br>
innerColor = (255, 255, 255)<br>
outerColor = (0, 0, 0)<br>
for y in range(imgsize[1]):<br>
    for x in range(imgsize[0]):<br>
        distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)<br>
        distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)<br>
        r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)<br>
        g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)<br>
        b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)<br>
        arr[y, x] = (int(r), int(g), int(b))<br>
plt.imshow(arr, cmap='gray')<br>
plt.show() <br>
 **output**<br>
![image](https://user-images.githubusercontent.com/98379636/181235153-34acc89c-7773-4bbc-8a77-3bb40b3433a1.png)<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>

imgsize=(650,650)<br>
image = Image.new('RGB', imgsize)<br>
innerColor = [153,0,0]<br>
for y in range(imgsize[1]):<br>
    for x in range(imgsize[0]):<br>
        distanceToCenter =np.sqrt((x - imgsize[0]/2) ** 2 + (y - imgsize[1]/2) ** 2)<br>
        distanceToCenter = (distanceToCenter) / (np.sqrt(2) * imgsize[0]/2)<br>
        r = distanceToCenter + innerColor[0] * (1 - distanceToCenter)<br>
        g = distanceToCenter + innerColor[1] * (1 - distanceToCenter)<br>
        b = distanceToCenter + innerColor[2] * (1 - distanceToCenter)<br>
        image.putpixel((x, y), (int(r), int(g), int(b)))<br>
plt.imshow(image)<br>
plt.show()<br>
**output** <br>                                                                                                                                     
 ![image](https://user-images.githubusercontent.com/98379636/181235312-6176beaa-7130-4484-9ac7-a7fb8d542970.png)<br>
   from PIL import Image<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
w, h = 512, 512<br>
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:100, 0:100] = [255, 0, 0]<br>
data[100:200, 100:200] = [255, 0, 255]<br>
data[200:300, 200:300] = [0, 255, 0]<br>
data[300:400, 300:400] = [255, 255, 0]<br>
data[400:500, 400:500] = [0, 255, 255]<br>
img = Image.fromarray(data, 'RGB')<br>
img.save('f1.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
 **output**   <br>                                                              
 ![image](https://user-images.githubusercontent.com/98379636/181235486-1919ed1a-5e28-4996-8f08-ee16050f886c.png)<br>
 
 # Python3 program for printing<br>
# the rectangular pattern<br>
 
# Function to print the pattern<br>
def printPattern(n):<br>
    arraySize = n * 2 - 1;<br>
    result = [[0 for x in range(arraySize)]<br>
                 for y in range(arraySize)];<br>   
    # Fill the values<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            if(abs(i - (arraySize // 2)) ><br>
               abs(j - (arraySize // 2))):<br>
                result[i][j] = abs(i - (arraySize // 2));<br>
            else:<br>
                result[i][j] = abs(j - (arraySize // 2));<br>
             
    # Print the array<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            print(result[i][j], end = " ");<br>
        print("");<br> 
# Driver Code<br>
n = 4;<br> 
printPattern(n);<br>
**output**<br>
3 3 3 3 3 3 3 
3 2 2 2 2 2 3 
3 2 1 1 1 2 3 
3 2 1 0 1 2 3 
3 2 1 1 1 2 3 
3 2 2 2 2 2 3 
3 3 3 3 3 3 3 
                                                                   
                                                                    































































