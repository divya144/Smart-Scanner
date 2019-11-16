import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import mapper
import imutils

def order_points(pts):

	rect = np.zeros((4, 2), dtype="float32")

	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]


	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect


def transformFourPoints(image, pts):

	rect = order_points(pts)
	(tl, tr, br, bl) = rect


	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))


	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))


	dst = np.array([[0, 0],	[maxWidth - 1, 0],	[maxWidth - 1, maxHeight - 1],	[0, maxHeight - 1]], dtype="float32")


	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped



img=cv.imread('img.jpeg',0)
img=cv.resize(img,(512,512))
orig=img.copy()
# plt.imshow(img,cmap='gray');plt.show()
blurred=cv.GaussianBlur(img,(5,5),0)
plt.imshow(blurred,cmap='gray');plt.show()
edged=cv.Canny(blurred,60,120)
plt.imshow(edged,cmap='gray');plt.show()
cnts,hierarchy=cv.findContours(edged,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)  
# contours=sorted(contours,key=cv.contourArea,reverse=True)
# print(len(contours))
# cnts = cnts[0] if imutils.is_cv() else cnts[1]  
cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]
cv.drawContours(img,cnts,0,(0,255,0),2)
plt.imshow(img,'gray');plt.show()
# peri = cv.arcLength(cnts[0], True)
# approx = cv.approxPolyDP(cnts[0], 0.02 * peri, True)

# cv.drawContours(img, [approx], -1, (0, 255, 0), 2)
# cv.imshow("Outline", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

warped = transformFourPoints(orig, cnts[0].reshape(4, 2) * ratio)

warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

print("STEP 3: Applying perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)