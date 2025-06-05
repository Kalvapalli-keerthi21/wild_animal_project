import cv2

cam=cv2.VideoCapture("http://192.0.0.4:8080/video")


while True:
    ret,frame=cam.read()

    cv2.imshow("result",frame)

    cv2.waitKey(1)