import cv2
import numpy as np
import Trim
import Model
def ImageToString(image):
    image = cv2.bitwise_not(image)

    num_labels, labeled_image = cv2.connectedComponents(image)
    s= [ ]
    for label in range(1, num_labels):
        component_mask = np.uint8(labeled_image == label) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        component_image = np.ones_like(image) * 255
        img,a = Trim.TrimImage(cv2.drawContours(component_image, contours, -1, (0, 0, 0), thickness=cv2.FILLED))
        s.append((Model.Recognition(img),a))
    s.sort(key=lambda x: x[1])
    s2 = ''
    for k,_ in s:
        s2+=k
    return s2