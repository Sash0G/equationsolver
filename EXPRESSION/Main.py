import Solve
import ImageSeparation
import cv2

img = cv2.imread(r'd:\GitHub\equationsolver\EQUATION\input.png', cv2.IMREAD_GRAYSCALE)
a = ImageSeparation.ImageToString(img)
print(a)
print(Solve.OPZ(a))
