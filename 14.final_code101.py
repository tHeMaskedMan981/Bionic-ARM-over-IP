import cv2
import numpy as np
import serial

ser = serial.Serial('COM4', 9600)


cap = cv2.VideoCapture(0)
counter1 =0
counter2 =0
counter3 =0
counter4 =0
counter5 =0
sum1,avg1=0,0
sum2,avg2=0,0
sum3,avg3=0,0
sum4,avg4=0,0
sum5,avg5=0,0



while True :
    a,frame = cap.read()
    hsv = cv2.cvtColor ( frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([20,100,100])
    upper_green = np.array([70,255,255])

    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    lower_red = np.array([150,150,50])
    upper_red = np.array([180,255,150])

    lower_skin=np.array([0,30,60])
    upper_skin=np.array([20,150,255])
	
    lower_magenta=np.array([80,100,100])
    upper_magenta=np.array([100,255,255])	
	
    mask1 = cv2.inRange(hsv, lower_green, upper_green)
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask3 = cv2.inRange(hsv, lower_red, upper_red)
    mask4 = cv2.inRange(hsv,lower_skin,upper_skin)
    mask5 = cv2.inRange(hsv,lower_magenta,upper_magenta)


    image1, contours1, hierarchy1 = cv2.findContours(mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    image2, contours2, hierarchy2 = cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    image3 , contours3,hierarchy3= cv2.findContours(mask3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    image4, contours4, hierarchy4 = cv2.findContours(mask4,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    image5, contours5, hierarchy5 = cv2.findContours(mask5,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
    
    print str(avg1)+"  "+str(avg2)+"  "+str(avg3)+"  "+str(avg4)+"  "+str(avg5)
    sum1+=len(contours1)
    counter1+=1
    if (counter1 ==10):
        avg1=sum1/10
        counter1=0
        sum1=0
        if avg1>200:
            ser.write('1')
        else :
            ser.write('0')
     
    sum2+=len(contours2)
    counter2+=1
    if (counter2 ==10):
        avg2=sum2/10
        counter2=0
        sum2=0
        if avg2>100:
            ser.write('3')
        else :
            ser.write('2')

     
    sum3+=len(contours3)
    counter3+=1
    if (counter3 ==10):
        avg3=sum3/10
        counter3=0
        sum3=0
        if avg3>100:
            ser.write('5')
        else :
            ser.write('4')  
    
    sum4+=len(contours4)
    counter4+=1
    if (counter4 ==10):
        avg4=sum4/10
        counter4=0
        sum4=0
        if avg4>500:
            ser.write('7')
        else :
            ser.write('6')
    
    sum5+=len(contours5)
    counter5+=1
    if (counter5 ==10):
        avg5=sum5/10
        counter5=0
        sum5=0
        if avg5>100:
            ser.write('9')
        else :
            ser.write('8')
    
    kernel = np.ones((5,5), np.uint8)
    
   # dilation1 = cv2.dilate(mask1, kernel, iterations=1)
    opening1 = cv2.morphologyEx( mask1, cv2.MORPH_OPEN, kernel)
    closing1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)

    #dilation2 = cv2.dilate(mask2, kernel, iterations=1)
    opening2 = cv2.morphologyEx( mask2, cv2.MORPH_OPEN, kernel)
    closing2 = cv2.morphologyEx(opening2, cv2.MORPH_CLOSE, kernel)

    #dilation3 = cv2.dilate(mask3, kernel, iterations=1)
    opening3 = cv2.morphologyEx( mask3, cv2.MORPH_OPEN, kernel)
    closing3 = cv2.morphologyEx(opening3, cv2.MORPH_CLOSE, kernel)

    opening4 = cv2.morphologyEx( mask4, cv2.MORPH_OPEN, kernel)
    closing4 = cv2.morphologyEx(opening4, cv2.MORPH_CLOSE, kernel)
  
    opening5 = cv2.morphologyEx( mask5,cv2.MORPH_OPEN, kernel)
    closing5 = cv2.morphologyEx(opening5, cv2.MORPH_CLOSE, kernel)
	
    result1 = cv2.bitwise_and(frame, frame, mask = closing1)
    result2 = cv2.bitwise_and(frame, frame, mask = closing2)
    result3 = cv2.bitwise_and(frame, frame, mask = closing3)
    result4 = cv2.bitwise_and(frame, frame, mask = closing4)
    result5 = cv2.bitwise_and(frame, frame, mask = closing5)  
    #median1 = cv2.medianBlur(result1,15)
    #median2 = cv2.medianBlur(result2,15)
    #median3 = cv2.medianBlur(result3,15)

    add = cv2.add(result1,result2)
    add1 = cv2.add(add, result3)
    add2 = cv2.add(add1, result4)
    add3 = cv2.add(add2, result5)	

    '''grayscale = cv2.cvtColor(add1, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny( grayscale, 80,100)
    result4 = cv2.bitwise_and(add1, add1, mask = edges)
    add2 = cv2.add(edges, result4)

    
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
        cv2.rectangle(add1, pt, (pt[0]+cols, pt[1]+rows), (0,255,255), 2)'''



    #cv2.imshow('median1',median1)
    #cv2.imshow('median2',median2)
    #cv2.imshow('median3',median3)
    cv2.imshow('add3',add3)
    #cv2.imshow('add2',add2)
    
    

    k=cv2.waitKey(5)
    if k==27:
        break


cap.release()
cv2.destroyAllWindows()
