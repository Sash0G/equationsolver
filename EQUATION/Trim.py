from PIL import Image
import numpy as np
from numpy import asarray
import cv2


def TrimImage(img):

    array = asarray(img)
    array = abs(array-255)
    rows = array.sum(axis=0)
    columns = array.sum(axis=1)
    newsize = [0, 0, 45, 45]
    # img = Image.fromarray(array)
    # img.save('geek.png')
    k = 0
    for i in range(len(rows)):
        if rows[i] != 0 and newsize[0] == 0:
            newsize[0] = i
        elif rows[i] == 0 and newsize[0] != 0:
            newsize[2] = i
            break
    for i in range(len(columns)):
        if columns[i] != 0 and newsize[1] == 0:
            newsize[1] = i
        elif columns[i] == 0 and newsize[1] != 0:
            newsize[3] = i
            break
    # print(img)
    img = cv2.bitwise_not(img)

    img = Image.fromarray(img)

    # newsize[0]-=5
    # newsize[2]+=5
    # newsize[1]-=5
    # newsize[3]+=5
    img = img.crop(newsize).resize([45, 45])
    # img.show()
    return (img, newsize[0])
