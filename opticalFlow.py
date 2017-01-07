import numpy as np
import cv2

#for lucas kanade method
lk_params = dict(winSize=(15,15),
                  maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#for feature detector
feature_params = dict(maxCorners=100,
                       qualityLevel=0.3,
                       minDistance=7,
                       blockSize=7)

path_to_video = "./data/cos.mov"
cap = cv2.VideoCapture(path_to_video)
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
mask = np.zeros_like(old_frame)

while(True):
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.line(mask, (a,b), (c,d), (0, 255, 0), 2)
        cv2.circle(frame,(a,b), 5, (0, 255, 0), -1)
    img = cv2.add(frame,mask)
    cv2.imshow('tracker', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    old_gray = frame_gray.copy()
cv2.destroyAllWindows()
cap.release()