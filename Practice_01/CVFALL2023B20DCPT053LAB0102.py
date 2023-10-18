import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

img = cv.imread('CVFALL2023B20DCPT053/CVFALL2023B20DCPT053002.jpg')
img_RGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
b = img_RGB.copy() # red = green = 0
b[:,:,1] =0
b[:,:,2] =0

g = img_RGB.copy() # red = blue = 0
g[:,:,0] =0
g[:,:,2] =0

r = img_RGB.copy() # blue = green = 0
r[:,:,0] =0
r[:,:,1] =0
cv.imshow('my img r',r)
cv.imshow('my img g',g)
cv.imshow('my img b',b)
cv.imshow('my img',img)
# hsv
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

cv.imshow('img hsv' , img_hsv)


h = img_hsv.copy()
h[:,:,1] =0
h[:,:,2] =0
cv.imshow('img hsv h' , h)

s = img_hsv.copy()
s[:,:,0] =0
s[:,:,2] =0
cv.imshow('img hsv s' , s)

v = img_hsv.copy()
v[:,:,0] =0
v[:,:,1] =0
cv.imshow('img hsv v' , v)

# show
r,g,b = cv.split(img_RGB)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
pixel_colors = img_RGB.reshape((np.shape(img_RGB)[0]*np.shape(img_RGB)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
# rgb
plt.show()

h,s,v = cv.split(img_hsv)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
# hsv
plt.show()

cmy_image = 255 - img
c,m,y = cv.split(cmy_image)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Cyan")
axis.set_ylabel("Magenta")
axis.set_zlabel("Yelow")
# cmyk
plt.show()
cv.waitKey(0)