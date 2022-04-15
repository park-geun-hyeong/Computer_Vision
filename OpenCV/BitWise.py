import cv2

img1 = cv2.imread('./bitwise1.png',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./bitwise2.png',cv2.IMREAD_GRAYSCALE)
img1 = cv2.resize(img1, (300,300))
img2 = cv2.resize(img2, (300,300))

AND = cv2.bitwise_and(img1, img2)
OR = cv2.bitwise_or(img1, img2)
XOR = cv2.bitwise_xor(img1,img2)
NOT = cv2.bitwise_not(img2)

print(img1.shape, img2.shape)

cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
cv2.imshow("AND",AND)
cv2.imshow("OR",OR)
cv2.imshow("XOR",XOR)
cv2.imshow("NOT",NOT)

cv2.waitKey(0)
cv2.destroyAllWindows()

