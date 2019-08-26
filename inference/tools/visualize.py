import cv2
import sys

# ColorMap for visualization more clearly
color_map = [[128, 64, 128], [244, 35, 231], [69, 69, 69], [102, 102, 156],
             [190, 153, 153], [153, 153, 153], [250, 170, 29], [219, 219, 0],
             [106, 142, 35], [152, 250, 152], [69, 129, 180], [219, 19, 60],
             [255, 0, 0], [0, 0, 142], [0, 0, 69], [0, 60, 100], [0, 79, 100],
             [0, 0, 230], [119, 10, 32]]

im = cv2.imread(sys.argv[1])

# Please note (224, 224) just for daheng model
print("visualizing...")
for i in range(0, 224):
    for j in range(0, 224):
        im[i, j] = color_map[im[i, j, 0]]

cv2.imwrite(sys.argv[1], im)
print("visualizing done!")
