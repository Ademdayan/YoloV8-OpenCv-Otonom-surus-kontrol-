import cv2
import numpy as np

########
# 1. OpenCV -> read a frame: cap.read()
# 2. Mouse Event -> define mouse event: cv2.EVENT_LBUTTONDOWN
# 3. Show Results
########

clicked_points = []
color = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX

path = "test_videos/road.mp4"
cap = cv2.VideoCapture(path)

ret, img = cap.read()
img = cv2.resize(img, (640,480))


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print("Coordinate:({},{})".format(x,y))
        cv2.circle(img, (x,y), 5, color, -1)
        cv2.putText(img, f'({x}, {y})', (x, y-10), font, 0.5, color, 2)

        cv2.imshow("Test", img)
        clicked_points.append((x,y))



cv2.imshow("Test", img)
cv2.setMouseCallback("Test", click_event)


if cv2.waitKey(0) == 27:
    cv2.imwrite("coordinates.png", img)

    for point in clicked_points:
        print(f"Coordinate: {point}")

cv2.destroyAllWindows()