import cv2
import numpy as np

image = cv2.imread('test.bmp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros_like(image)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)

result = cv2.bitwise_and(image, mask)

cropped_result = result[y:y+h, x:x+w]

cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('result_crop.png', cropped_result)
cv2.imwrite('result.png', result)
