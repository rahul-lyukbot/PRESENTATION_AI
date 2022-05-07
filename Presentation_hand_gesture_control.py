import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np


# First thing we have to get our frames from camera for this we use cv2
# here we define the height and width of because we use in widow size
width = 1280
height = 720

cap = cv2.VideoCapture(0)     #Here 0 is the ID no of webcam you can change it if you have multiple camera connected to your machine
cap.set(3, width)
cap.set(4, height)
# Here we describe the path of our presentation folder
folder_path = "C:\\Users\\Rahul\\PycharmProjects\\AI_Project\\Testing_resources\\P2"


# Getting the list of image of our presentation
list_pre = sorted(os.listdir(folder_path), key=len)  # Here we are sort our list of image because if we have more than 10 image it seems like a problem

# Here we define our detector for hand detection
detector = HandDetector(detectionCon=0.8, maxHands=1)


# variables-> these are the variables which we use further
img_number = 0
gesture_threshold = 300
button_press = False
button_counter = 0
button_delay = 10
annotations = [[]]  # For drawing line's between points
annotation_number = -1
annotation_start = False

# Now setting height and width for our webcam image
h_small, w_small = int(120*1), int(213*1)


while True:
    success, img = cap.read()
    # Because it's show us a mirror image we need to flip it for our convince
    img = cv2.flip(img, 1)
    # Accessing our Presentation images
    path_image = os.path.join(folder_path, list_pre[img_number])
    current_img = cv2.imread(path_image)
    # Adding webcam image onto our presentation sildes
    small_img = cv2.resize(img, (w_small, h_small))
    h, w, _ = current_img.shape
    current_img[0:h_small, w-w_small:w] = small_img
    # Showing the detected hands on our image
    hands, img = detector.findHands(img)
    # here we draw a line for indicates the points where the gesture works
    cv2.line(img, (0, gesture_threshold), (width, gesture_threshold), (0, 255, 0), 5)
    if hands and button_press is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand["center"]
        # for adding more gesture we need to find the position of our index and middle fingers
        lmList = hand["lmList"]       # lmlist  is stand for land_mark list
        # constraint the draw area for easy for drawings for this we use numpy
        x_value = int(np.interp(lmList[8][0], [width/2, w], [0, w]))
        y_value = int(np.interp(lmList[8][1], [150, height-150], [0, h]))
        index_finger = x_value, y_value
        # Adding gesture
        if cy <= gesture_threshold:
            # Gesture 1 -> Left
            if fingers == [1, 0, 0, 0, 0]:
                annotation_start = False
                if img_number > 0:
                    button_press = True
                    annotations = [[]]
                    annotation_number = 0
                    img_number -= 1

            # Gesture 2 -> Right
            if fingers == [0, 0, 0, 0, 1]:
                annotation_start = False
                if img_number < len(path_image) - 1:
                    button_press = True
                    annotations = [[]]
                    annotation_number = 0
                    img_number += 1

        # Gesture 3 -> Pointer
        '''->we need to take out this gesture from our main if statement because we need pointer everywhere
          not only the above part of our presentation'''
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(current_img, index_finger, 12, (0, 0, 255), cv2.FILLED)
            annotation_start = False

        # Gesture 4 -> draw
        if fingers == [0, 1, 0, 0, 0]:
            if annotation_start is False:
                annotation_start = True
                annotation_number += 1
                annotations.append([])
            cv2.circle(current_img, index_finger, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotation_number].append(index_finger)
        else:
            annotation_start = False

        # Gesture 5 -> Erase
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotation_number -= 1
                button_press = True

    # Iteration for button press
    if button_press:
        button_counter += 1
        if button_counter > button_delay:
           button_counter = 0
           button_press = False

    # Draw the line between our annotations points of our index fingers
    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(current_img, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 12)

    cv2.imshow("Image", img)
    cv2.imshow("Slide", current_img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break