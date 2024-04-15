from PIL import Image
import numpy as np
from numpy import asarray

img = Image.open('img.png')
array = asarray(img)
array = abs(array-255)
array2D = array.sum(axis=2)
rows = array2D.sum(axis=0)
columns = array2D.sum(axis=1)
newsize = np.full((1000,4),(0,0,45,45))
print(rows)
print(columns)
k=0
for i in range(len(rows)):
    if rows[i] != 0 and newsize[k][0] == 0:
       newsize[k][0] = i
    elif rows[i] == 0 and newsize[k][0] != 0:
        newsize[k][2] = i
        k+=1
print(k)
for i in range(len(columns)):
    if columns[i] != 0 and newsize[0][1] == 0:
        for j in range(k): newsize[j][1] = i
    elif columns[i] == 0 and newsize[0][1] != 0:
        for j in range(k): newsize[j][3] = i
        break
print(newsize)
for i in range(k):
    img.crop(newsize[i]).resize([45,45]).save("./test/j"+str(i)+".png")

