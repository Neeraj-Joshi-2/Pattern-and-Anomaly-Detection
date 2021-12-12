"""

import cv2

path = r'D:\DOWNLOADS\image1.jpg'

img = cv2.imread(path)
  
cv2.imshow('image', img)

"""

"""


from PIL import Image 
import numpy as np

img = Image.open(r"D:\DOWNLOADS\image1.jpg")

imgArray = np.array(img)

print(type(img))
print(imgArray)

img.show()

"""

"""
from PIL import Image 
import matplotlib.pyplot as plt

img = Image.open('D:\DOWNLOADS\image1.jpg')

plt.imshow(img)

print('here')

"""

from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt

img = Image.open(r"D:\DOWNLOADS\image1.jpg")

imgArray = np.array(img)

print(type(img))
print(imgArray)

plt.imshow(img)