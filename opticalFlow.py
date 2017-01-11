import cv2

#PRESS S TO START TRACKING

#For lucas kanade method
lk_params = dict(winSize=(15,15),
                  maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#For feature detector
feature_params = dict(maxCorners=100,
                       qualityLevel=0.3,
                       minDistance=7,
                       blockSize=7)

path_to_video = "./3031210.mp4"
file_name = "detected.txt"

f = open(file_name, 'r')
coords = f.read().split(' ')
coords = [coord.rstrip() for coord in coords]

cap = cv2.VideoCapture(path_to_video)
cv2.namedWindow('Main')

while(1):
    while(1):
        _, frame = cap.read()
        cv2.rectangle(frame, (int(float(coords[0])), int(float(coords[1]))),
                      (int(float(coords[2])), int(float(coords[3]))),
                      color=(100, 255, 100), thickness=4)

        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #ROI
        roi = frameGray[int(float(coords[1])):int(float(coords[3])),
              int(float(coords[0])):int(float(coords[2]))]

        #Feature detector
        new_corners = cv2.goodFeaturesToTrack(roi, **feature_params)
        new_corners[:, 0, 0] = new_corners[:, 0, 0] + int(float(coords[0]))
        new_corners[:, 0, 1] = new_corners[:, 0, 1] + int(float(coords[1]))

        for corner in new_corners:
            cv2.circle(frame, (int(corner[0][0]), int(corner[0][1])), 5, (255, 0, 0))

        oldFrameGray = frameGray.copy()
        old_corners = new_corners.copy()

        cv2.imshow('Main', frame)

        k = 0xFF & cv2.waitKey(5)
        if k == 27:
            cv2.destroyAllWindows()
            cap.release()
        elif k == ord('s'):
            break

    while(1):
        ret, frame = cap.read()
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Optical flow
        new_corners, p0, p1 = cv2.calcOpticalFlowPyrLK(oldFrameGray, frameGray,
                                                        old_corners, None, **lk_params)
        new_corners_updated = new_corners.copy()
        for corner in new_corners_updated:
            cv2.circle(frame, (int(corner[0][0]), int(corner[0][1])), 5, (0, 255, 0))

        oldFrameGray = frameGray.copy()
        old_corners = new_corners_updated.copy()

        cv2.imshow('Main', frame)

        k = 0xFF & cv2.waitKey(5)
        if k == 27:
            cv2.destroyAllWindows()
            cap.release()
        elif k == ord('s'):
            break

cv2.destroyAllWindows()
cap.release()