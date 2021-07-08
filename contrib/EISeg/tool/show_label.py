import cv2
import matplotlib.pyplot as plt

img = cv2.imread("/home/lin/Desktop/label/467594346.png")
img = img * 255
plt.imshow(img)
plt.show()
