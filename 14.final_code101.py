import cv2
import numpy as np
import serial

ser = serial.Serial('COM4', 9600)
ser.write('1')

cap = cv2.VideoCapture(0)
counter =0
sum=0

while True :
    a,frame = cap.read()
    hsv = cv2.cvtColor ( frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([20,100,100])
    upper_green = np.array([70,255,255])

    lower_blue = np.array([90,20,20])
    upper_blue = np.array([150,255,255])

    lower_red = np.array([150,150,0])
    upper_red = np.array([180,255,255])

    mask1 = cv2.inRange(hsv, lower_green, upper_green)
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask3 = cv2.inRange(hsv, lower_red, upper_red)

    image1, contours1, hierarchy1 = cv2.findContours(mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    image2, contours2, hierarchy2 = cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    
    print len(contours1)
    sum+=len(contours1)
    counter+=1
    if (counter ==5):
        avg=sum/5
        counter=0
        sum=0
        if avg>100:
            ser.write('1')

        else :
            ser.write('0')

    #print 'blue' + str(len(contours2))
    #if len(contours2)>600:
     #   ser.write('2')
    #else :
     #   ser.write('3')

   
        
    
    kernel = np.ones((5,5), np.uint8)
    
    dilation1 = cv2.dilate(mask1, kernel, iterations=1)
    opening1 = cv2.morphologyEx( dilation1, cv2.MORPH_OPEN, kernel)
    closing1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)

    #dilation2 = cv2.dilate(mask2, kernel, iterations=1)
    opening2 = cv2.morphologyEx( mask2, cv2.MORPH_OPEN, kernel)
    closing2 = cv2.morphologyEx(opening2, cv2.MORPH_CLOSE, kernel)

    #dilation3 = cv2.dilate(mask3, kernel, iterations=1)
    opening3 = cv2.morphologyEx( mask3, cv2.MORPH_OPEN, kernel)
    closing3 = cv2.morphologyEx(opening3, cv2.MORPH_CLOSE, kernel)

    result1 = cv2.bitwise_and(frame, frame, mask = closing1)
    result2 = cv2.bitwise_and(frame, frame, mask = closing2)
    result3 = cv2.bitwise_and(frame, frame, mask = closing3)
    
    #median1 = cv2.medianBlur(result1,15)
    #median2 = cv2.medianBlur(result2,15)
    #median3 = cv2.medianBlur(result3,15)

    add = cv2.add(result1,result2)
    add1 = cv2.add(add, result3)

    '''grayscale = cv2.cvtColor(add1, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny( grayscale, 80,100)
    result4 = cv2.bitwise_and(add1, add1, mask = edges)
    add2 = cv2.add(edges, result4)'''

    
    img1gray = cv2.cvtColor(add1, cv2.COLOR_BGR2GRAY)
    tempgreen = cv2.imread('templategreen.jpg',0)
    tempblue = cv2.imread('templateblue.jpg',0)

    rows, cols = tempgreen.shape                        
    w,h = tempblue.shape[::-1]

    resgreen = cv2.matchTemplate(img1gray, tempgreen, cv2.TM_CCOEFF_NORMED)
    resblue = cv2.matchTemplate(img1gray, tempblue, cv2.TM_CCOEFF_NORMED)
    #ret2,threshold2 = cv2.threshold(res2, 8, 255, cv2.THRESH_BINARY)

    threshold =0.8
    threshold3 =0.8
                        
    loc = np.where(resgreen>=threshold)
    loc2 = np.where(resblue>=threshold3)
    
    for pt in zip(*loc[::-1]) :
        cv2.rectangle(add1, (pt[0],pt[1]-50), (pt[0]+w, pt[1]+h-10), (0,255,255), 2)

    for pt in zip(*loc2[::-1]) :
        cv2.rectangle(add1, pt, (pt[0]+cols, pt[1]+rows), (0,255,255), 2)



    #cv2.imshow('median1',median1)
    #cv2.imshow('median2',median2)
    #cv2.imshow('median3',median3)
    cv2.imshow('add1',add1)
    #cv2.imshow('add2',add2)
    
    

    k=cv2.waitKey(5)
    if k==27:
        break


cap.release()
cv2.destroyAllWindows()
