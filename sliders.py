import cv2
import numpy as np

def mask_hsv(img, h_min, h_max, s_min, s_max, v_min, v_max):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a mask for the specified HSV ranges
    mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))

    # Apply the mask to the original image
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return masked_img

def main():
    # Load the image
    img = cv2.imread('p.png')

    # Create sliders for HSV ranges
    cv2.namedWindow('HSV Mask')
    cv2.createTrackbar('H Min', 'HSV Mask', 0, 179, lambda x: None)
    cv2.createTrackbar('H Max', 'HSV Mask', 179, 179, lambda x: None)
    cv2.createTrackbar('S Min', 'HSV Mask', 0, 255, lambda x: None)
    cv2.createTrackbar('S Max', 'HSV Mask', 255, 255, lambda x: None)
    cv2.createTrackbar('V Min', 'HSV Mask', 0, 255, lambda x: None)
    cv2.createTrackbar('V Max', 'HSV Mask', 255, 255, lambda x: None)

    while True:
        # Get slider values
        h_min = cv2.getTrackbarPos('H Min', 'HSV Mask')
        h_max = cv2.getTrackbarPos('H Max', 'HSV Mask')
        s_min = cv2.getTrackbarPos('S Min', 'HSV Mask')
        s_max = cv2.getTrackbarPos('S Max', 'HSV Mask')
        v_min = cv2.getTrackbarPos('V Min', 'HSV Mask')
        v_max = cv2.getTrackbarPos('V Max', 'HSV Mask')

        # Apply the mask
        masked_img = mask_hsv(img, h_min, h_max, s_min, s_max, v_min, v_max)

        # Display the masked image
        cv2.imshow('HSV Mask', masked_img)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()