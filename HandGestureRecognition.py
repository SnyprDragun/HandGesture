import cv2
import numpy as np
import math

#WebCam Pointer
cap=cv2.VideoCapture(0)

#Loop for recording input as long as WebCam is open
while (cap.isOpened()):
    ret, img=cap.read()

    #SubWindow Dimensions
    cv2.rectangle(img, (800, 800), (800, 800), (0, 255, 0), 0)
    crop_img=img[100:600, 100:600]

    #Conversion from RGB to greyscale then to binary in order to find ROI (Region of Interest)
    grey=cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    #applying Gaussian Blur to image = low pass filter (not interested in details but shape of image)
    value=(35, 35)
    blurred=cv2.GaussianBlur(grey, value, 0)

    #thresholding to give black background and white image shape
    _, thresh1=cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #shows thresholded image
    cv2.imshow('Thresholded', thresh1)

    #checking OpenCV unpacking error
    #(version, _, _)=cv2._version_.split('.')

    #if version=='3':
    #image, contours, hierarchy=cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #elif version=='4':
    contours, hierarchy=cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    #finding contours
    cnt=max(contours, key=lambda x: cv2.contourArea(x))

    #bounding rectangle around Contour
    x, y, w, h=cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    #ConvexHull=tight fitting convex boundary around the points or the shape
    hull=cv2.convexHull(cnt)

    #drawing contours
    drawing=np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 255, 0), 0)

    #finding convex hull
    hull=cv2.convexHull(cnt, returnPoints=False)

    #finding convexity defects
    defects=cv2.convexityDefects(cnt, hull)
    count_defects=0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    #applying cosine rule to find angle b/w fingers
    for i in range(defects.shape[0]):
        s, e, f, d=defects[i,0]
        
        #finding dimensions of defects/fingers
        start=tuple(cnt[s][0])
        end=tuple(cnt[e][0])
        far=tuple(cnt[f][0])

        #finding length of all sides of triangle
        a=math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
        b=math.sqrt((far[0]-start[0])**2+(far[1]-start[1])**2)
        c=math.sqrt((end[0]-far[0])**2+(end[1]-far[1])**2)

        #cosine rule for angle
        angle=math.acos((b**2+c**2-a**2)/(2*b*c))*57

        #if angle<90 then we count a finger and draw red circle around it
        if angle<=90:
            count_defects+=1
            cv2.circle(crop_img, far, 1, [0, 0, 255], -1)
        
        cv2.line(crop_img, start, end, [0, 255, 0], 2)

    #counting fingers
    if count_defects==0:
        cv2.putText(img, "We detect one finger", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    if count_defects==1:
        cv2.putText(img, "We detect two finger", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects==2:
        cv2.putText(img, "We detect three fingers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects==3:
        cv2.putText(img, "TWe detect four fingers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects==4:
        cv2.putText(img, "We detect five fingers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    #else:
        #cv2.putText(img, "TWe detect a hand", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    
    cv2.imshow('Gesture', img)
    all_img=np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)

    #escape key break the loop and 27 represents escape key
    k=cv2.waitKey(10)
    if k==27:
        break


cap.release()
cv2.destroyAllWindows()
