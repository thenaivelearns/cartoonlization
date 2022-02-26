import cv2
import numpy as np

# Reading the Image 
image = cv2.imread("3.jpg")

# 第一次变换（卡通化）
# Finding the Edges of Image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
gray = cv2.medianBlur(gray, 7) 
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
# Making a Cartoon of the image
color = cv2.bilateralFilter(image, 12, 250, 250) 
cartoon = cv2.bitwise_and(color, color, mask=edges)
#Visualize the cartoon image 
cv2.imshow("Cartoon1", cartoon)
cv2.imwrite('Cartoon1.jpg',cartoon)
cv2.waitKey(0) # "0" is Used to close the image window
cv2.destroyAllWindows()
#cv2.imshow("edges",edges)


#第二次变换（模糊图像）
#convert to gray scale
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#apply gaussian blur
grayImage = cv2.GaussianBlur(grayImage, (3, 3), 0)
#detect edges
edgeImage = cv2.Laplacian(grayImage, -1, ksize=3)
edgeImage = 255 - edgeImage
#threshold image
ret, edgeImage = cv2.threshold(edgeImage, 150, 255, cv2.THRESH_BINARY)
#blur images heavily using edgePreservingFilter
edgePreservingImage = cv2.edgePreservingFilter(image, flags=2, sigma_s=50, sigma_r=0.4)
#create output matrix
output =np.zeros(grayImage.shape)
#combine cartoon image and edges image
output = cv2.bitwise_and(edgePreservingImage, edgePreservingImage, mask=edgeImage)
#Visualize the cartoon image 
cv2.imshow("Cartoon2", output)
cv2.imwrite('Cartoon2.jpg',output)
cv2.waitKey(0) # "0" is Used to close the image window
cv2.destroyAllWindows()
#cv2.imshow("edgeImage",edgeImage)


#第三次变换（风格化）
cartoon_image = cv2.stylization(image, sigma_s=150, sigma_r=0.45)  
cv2.imshow('Cartoon3', cartoon_image)
cv2.imwrite('Cartoon3.jpg',cartoon_image)
cv2.waitKey(0)  
cv2.destroyAllWindows()


#第四次变换（铅笔素描）
cartoon_image1, cartoon_image2  = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.15, shade_factor=0.05)  
cv2.imshow('pencil1', cartoon_image1)
cv2.imwrite('pencil1.jpg',cartoon_image1)
cv2.waitKey()  
cv2.destroyAllWindows()

cv2.imshow('pencil2', cartoon_image2)
cv2.imwrite('pencil2.jpg',cartoon_image2)
cv2.waitKey()    
cv2.destroyAllWindows()



