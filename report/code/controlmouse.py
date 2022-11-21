import numpy as np
import cv2
from collections import deque
import numpy as np
from pymouse import PyMouse


def dist(a,b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5

def fingerCursor(device):
    cap = cv2.VideoCapture(device)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    ## skin color segmentation mask
    skin_min = np.array([0, 70, 50],np.uint8)  # HSV mask
    skin_max = np.array([30, 160, 255],np.uint8) # HSV mask

    ## trajectory drawing initialization
    topmost_last = (200,100)    # initial position of finger cursor
    traj = np.array([], np.uint16)
    traj = np.append(traj, topmost_last)
    dist_pts = 0
    dist_records = [dist_pts]

    ## finger cursor position low_pass filter
    low_filter_size = 5
    low_filter = deque([topmost_last,topmost_last,topmost_last,topmost_last,topmost_last],low_filter_size )  # filter size is 5

    ## gesture matching initialization
    gesture2 = cv2.imread("./gesture.png")
    gesture2 = cv2.cvtColor(gesture2, cv2.COLOR_BGR2GRAY)
    gesture2 , _ = cv2.findContours(gesture2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    ## gesture_index low_pass filter
    gesture_filter_size = 5
    gesture_matching_filter = deque([0.0,0.0,0.0,0.0,0.0], gesture_filter_size )
    gesture_index_thres = 0.8

    ## color definition
    green = (0,255,0)
    red = (0,0,255)

    ## background segmentation
    # some kernels
    kernel_size = 5
    kernel1 = np.ones((kernel_size,kernel_size),np.float32)/kernel_size/kernel_size
    
    m = PyMouse()

    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, frame_raw = cap.read()
        while not ret:
            ret,frame_raw = cap.read()
        frame_raw = cv2.flip(frame_raw,1)
        frame = frame_raw[:round(cap_height),:round(cap_width)]    # ROI of the image
        cv2.imshow('raw_frame',frame)

        # Color seperation and noise cancellation at HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, skin_min, skin_max)
        res = cv2.bitwise_and(hsv, hsv, mask= mask)
        res = cv2.erode(res, kernel1, iterations=1)
        res = cv2.dilate(res, kernel1, iterations=1)

        # Canny edge detection at Gray space.
        rgb = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        cv2.imshow('rgb_2',rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        cv2.imshow('gray',gray)

        ## main function: find finger cursor position & draw trajectory
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    # find all contours in the image
        if len(contours) !=0:
            c = max(contours, key = cv2.contourArea)  # find biggest contour in the image
            if cv2.contourArea(c) > 1000:
                topmost = tuple(c[c[:,:,1].argmin()][0])  # consider the topmost point of the biggest contour as cursor
                gesture_index = cv2.matchShapes(c,gesture2[0],2,0.0)

                # obtain gesture matching index using gesture matching low_pass filter
                gesture_matching_filter.append(gesture_index)
                sum_gesture = 0
                for i in gesture_matching_filter:
                    sum_gesture += i
                gesture_index = sum_gesture/gesture_filter_size
                print(gesture_index)

                for i in contours:
                    if cv2.contourArea(c) <= 50:
                        continue
                    x, y, _, _ = cv2.boundingRect(c)
                    m.move(x, y)

                dist_pts = dist(topmost,topmost_last)  # calculate the distance of last cursor position and current cursor position
                if dist_pts < 150:  # filter big position change of cursor
                    try:
                        low_filter.append(topmost)
                        sum_x = 0
                        sum_y = 0
                        for i in low_filter:
                            sum_x += i[0]
                            sum_y += i[1]
                        topmost = (sum_x//low_filter_size, sum_y//low_filter_size)

                        if gesture_index > gesture_index_thres:
                            traj = np.append( traj, topmost)
                            dist_records.append(dist_pts)
                            
                        else:
                            traj = np.array([], np.uint16)
                            traj = np.append(traj, topmost_last)
                            dist_pts = 0
                            dist_records = [dist_pts]
                            pass
                        topmost_last = topmost  # update cursor position
                    except:
                        print('error')
                        pass
              
        ## drawing trajectory in video:(if just want to move the mouse, # the following 4 lines.)
        for i in range(1, len(dist_records)):
            thickness = int(-0.072 * dist_records[i] + 13)
            cv2.line(frame, (traj[i*2-2],traj[i*2-1]), (traj[i*2],traj[i*2+1]), red , thickness)
            cv2.line(rgb, (traj[i*2-2],traj[i*2-1]), (traj[i*2],traj[i*2+1]), red , thickness)
           
        cv2.circle(frame, topmost_last, 10, green , 3)
        cv2.circle(rgb, topmost_last, 10, green , 3)
        # moving the mouse(if just want to draw trajectory, # the next line.)
        m.move(topmost_last[0], topmost_last[1])

        ## Display the resulting frame
        cv2.imshow('rgb', rgb)
        cv2.imshow('frame', frame_raw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ## When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    device = 0    # if device = 0, use the built-in computer camera
    fingerCursor(device)