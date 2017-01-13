import cv2
import os

#For lucas kanade method
lk_params = dict(winSize=(15,15),
                  maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#For feature detector
feature_params = dict(maxCorners=100,
                       qualityLevel=0.3,
                       minDistance=7,
                       blockSize=7)

path_to_video = "./data/new_two_sheep/sheep.mp4"
file_name = "./data/new_two_sheep/detected"

list_sheet = []
frame_count = 0

cap = cv2.VideoCapture(path_to_video)
cv2.namedWindow('Tracker')

dir_work = os.listdir("./data/new_two_sheep")
count_files = len(dir_work) - 1

class Sheep:
    def __init__(self, list):
        self.coords = list
        self.roi = []
        self.new_corners = []
        self.old_corners = []
        self.new_corners_updated = []

def read_coords(n):
    f = open(file_name + str(n) + ".txt", 'r')
    for line in f.readlines():
        coords = line.split(' ')
        coords = [coord.rstrip() for coord in coords]
        s = Sheep(coords)
        list_sheet.append(s)
    f.close()

while(count_files != 0):
    while(1):
        read_coords(frame_count)
        _, frame = cap.read()
        frame_count += 1
        for st in list_sheet:
            cv2.rectangle(frame, (int(float(st.coords[0])), int(float(st.coords[1]))),
                          (int(float(st.coords[2])), int(float(st.coords[3]))),
                          color=(100, 255, 100), thickness=4)

        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ROI
        for st in list_sheet:
            st.roi = frameGray[int(float(st.coords[1])):int(float(st.coords[3])),
                     int(float(st.coords[0])):int(float(st.coords[2]))]

        # Feature detector
        for st in list_sheet:
            st.new_corners = cv2.goodFeaturesToTrack(st.roi, **feature_params)
            st.new_corners[:, 0, 0] = st.new_corners[:, 0, 0] + int(float(st.coords[0]))
            st.new_corners[:, 0, 1] = st.new_corners[:, 0, 1] + int(float(st.coords[1]))

            for corner in st.new_corners:
                cv2.circle(frame, (int(corner[0][0]), int(corner[0][1])), 5, (255, 0, 0))
            st.old_corners = st.new_corners.copy()

        oldFrameGray = frameGray.copy()

        cv2.imshow('Tracker', frame)
        break
    while(1):
        ret, frame = cap.read()
        frame_count += 1
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical flow
        for st in list_sheet:
            st.new_corners, p0, p1 = cv2.calcOpticalFlowPyrLK(oldFrameGray, frameGray,
                                                       st.old_corners, None, **lk_params)
            st.new_corners_updated = st.new_corners.copy()

        for st in list_sheet:
            for corner in st.new_corners_updated:
                cv2.circle(frame, (int(corner[0][0]), int(corner[0][1])), 5, (0, 255, 0))

        # Drawing rectangle
        for st in list_sheet:
            x, y, w, h = cv2.boundingRect(st.new_corners_updated)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        oldFrameGray = frameGray.copy()

        for st in list_sheet:
            st.old_corners = st.new_corners_updated.copy()

        cv2.imshow('Tracker', frame)
        if (frame_count %  50 == 0):
    #      list_sheet = []
            count_files -= 1
            break

        k = 0xFF & cv2.waitKey(50)
        if k == 27:
            cv2.destroyAllWindows()
            cap.release()
        elif k == ord('s'):
            break

cv2.destroyAllWindows()
cap.release()