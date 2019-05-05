import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
def faceDetect():
    eye_detect = False
    face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    info = ''
    
    try:
        #cap = cv2.VideoCapture('images/group.jpg')
        cap = cv2.imread('images/group.jpg')
    except:
        print('카메라 로딩 실패')
        return
    frame = cap
        
    if eye_detect:
        info = 'Eye Detection On'
    else:
        info = 'Eye Detection Off'
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    
    num = len(faces)
        
    if num > 0:
        (x1, y1, w1, h1) = faces[0]
        a = frame[y1:y1+h1,x1:x1+w1]
        a = cv2.resize(a, (150,150))
            #cv2.imshow('a',a)
    if num > 1:
        (x2, y2, w2, h2) = faces[1]
        b = frame[y2:y2+h2,x2:x2+w2]
        b = cv2.resize(b, (150,150))
        #cv2.imshow('b',b)   
        
        a, b = b, a
        a = cv2.resize(a,(h1,w1))
        b = cv2.resize(b,(h2,w2))
        frame[y1:y1+h1,x1:x1+w1] = a
        frame[y2:y2+h2,x2:x2+w2] = b

        
    cv2.putText(frame, info, (5,15),font, 0.5, (255,0,255),1)
        
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame, 'Detected Face', (x-5,y-5),font, 0.5,(255,255,0),2)
        if eye_detect:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for(ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    
    cv2.imshow('frame',frame)
        
        #out.write(frame)
    cv2.imwrite('images/output.jpg',frame)
        
    k = cv2.waitKey(30)
    if k == ord('i'):
        eye_detect = not eye_detect
        
    cv2.waitKey(0)
    cap.release()
    
    cv2.destroyAllWindows()
    
faceDetect()
