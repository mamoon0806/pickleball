import cv2
import numpy as np

path = r"D:/pythonprojects/pickleball/pickleball.mp4"

cap = cv2.VideoCapture(path)

frame_rate = 10
cap.set(cv2.CAP_PROP_FPS, frame_rate)

yellow_lower = np.array([18, 146, 208])
yellow_upper = np.array([51, 255, 255])

ret, frame = cap.read()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while(cap.isOpened()):
    ret, frame = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    masked_flow = flow * mask[..., None]
    mag, ang = cv2.cartToPolar(masked_flow[..., 0], masked_flow[..., 1])
    threshold = 150  # Adjust threshold as needed
    motion_mask = mag > threshold

    print(np.sum(motion_mask))

    if np.sum(motion_mask) > 0:
        print("Possible bounce detected!")

    prev_gray = gray

    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if len(contours) > 0:
    #     largest_contour = max(contours, key=cv2.contourArea)
        
    #     M = cv2.moments(largest_contour)
    #     if M["m00"] != 0:
    #         cx = int(M["m10"] / M["m00"])
    #         cy = int(M["m01"] / M["m00"])

    #         # Draw a circle around the ball
    #         cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()