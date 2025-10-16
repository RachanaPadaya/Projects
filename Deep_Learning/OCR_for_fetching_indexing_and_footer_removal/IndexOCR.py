import cv2
from PIL import Image
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

img_file = 'Data/index_02.JPG'
img = cv2.imread(img_file) 
base_img = img.copy()

img = cv2.imread(img_file)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur_img = cv2.GaussianBlur(gray_img, (7,7), 0) #add gaussian blur
thresh_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernel_img = cv2.getStructuringElement(cv2.MORPH_RECT, (3,13)) #define kernel , blur long horoizontal and thin vertical
dilate = cv2.dilate(thresh_img, kernel_img, iterations=1) #apply dilation

cnts_img = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #find contours
cnts_img = cnts_img[0] if len(cnts_img) == 2 else cnts_img[1]  #get contours
cnts_img = sorted(cnts_img, key=lambda x: cv2.boundingRect(x)[0])   #sort contours from left to right

i = 0
results = []
for c in cnts_img: 
    x,y,w,h = cv2.boundingRect(c)
    if h>200 and w>20: #filtering small contours 
        roi = img[y:y+h, x:x+w] #crop roi
        # cv2.imwrite('Output/roi'+str(i)+'.png', roi)
        # i +=1
        cv2.rectangle(img, (x,y), (x+w, y+h), (36,255,12), 2) #draw rectangle
        ocr_result = pytesseract.image_to_string(roi)
        # print(ocr_result)
        ocr_result = ocr_result.split('\n')
        for word in ocr_result:
           results.append(word)

cv2.imwrite('Output/bbox_img.png', img)
print(results)

entities = []
for item in results:
    item = item.strip().replace('\n', ' ')
    item = item.split(' ')[0]
    if len(item)>2:
        if item[0] == 'A' and '-' not in item:
            item = item.split('.')[0].replace(',','').replace('\'','').replace('"','').replace('`','').replace('‘','').replace('’','').replace('“','').replace('”','').replace('·','').replace('•','').replace('●','').replace('◦','').replace('▪','').replace('▫','').replace('–','-').replace('—','-').strip()
            entities.append(item)
entities = list(set(entities))
entities.sort()
print(entities)
