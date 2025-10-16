import cv2
from PIL import Image
import pytesseract
import numpy as np

image = cv2.imread('Data/sample_mgh.JPG')
base_img = image.copy()
im_h, im_w, im_d = image.shape

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
blur_img = cv2.GaussianBlur(gray_img, (7,7), 0) #add gaussian blur
thresh_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

#create rectangular kernel and dilate
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,10)) #define kernel , blur long horoizontal and thin vertical
dilate_img = cv2.dilate(thresh_img, kernel, iterations=1) #apply dilation

#fibd contours and draw rectangles
cnts = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if h < 20 and w > 250: 
        roi = base_img[0:y+h, 0:x+im_w] # modify to remove everything below the line
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    
cv2.imwrite('Output/output.png', roi)    #save the cropped image