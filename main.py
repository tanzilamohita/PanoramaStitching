# importing libraries
import cv2
import numpy as np

# read input image
img_ = cv2.imread('2.JPG')
# img_ = cv2.imread('Image2.png')
img_ = cv2.resize(img_, (0, 0), fx=1, fy=1)
img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
img = cv2.imread('1.jpg')
# img = cv2.imread('Image1.png')
img = cv2.resize(img, (0, 0), fx=1, fy=1)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step1. Keypoints detection using SIFT
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
cv2.imshow('original_image_1_keypoints', cv2.drawKeypoints(img_, kp1, None))
cv2.imshow('original_image_2_keypoints', cv2.drawKeypoints(img, kp2, None))

# Step2. Descriptors matching between two images using FLANNBasedMatcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
match = cv2.FlannBasedMatcher(index_params, search_params)
matches = match.knnMatch(des1, des2, k=2)

# Step2. Descriptors matching between two images using BFMatcher
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m in matches:
    if m[0].distance < 0.5*m[1].distance:
         good.append(m)
    matches = np.asarray(good)

draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
img3 = cv2.drawMatchesKnn(img_, kp1, img, kp2, good, None, **draw_params)
cv2.imshow("original_image_drawMatches.png", img3)

# Step3. Homography estimation using matched feature vectors
if len(matches[:, 0]) >= 4:
    src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    print(H)
else:
    raise AssertionError('Can’t find enough keypoints.')

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    #crop right
    elif not np.sum(frame[:, -1]):
        return trim(frame[:, :-12])
    return frame

# Step4. Warping transformation using homography
dst = cv2.warpPerspective(img_, H, (img.shape[1] + img_.shape[1], img.shape[0]))
cv2.imshow("Warped Image", dst)
dst[0:img.shape[0], 0:img.shape[1]] = img
cv2.imshow("output.jpg", trim(dst))
cv2.imwrite("output.jpg", trim(dst))
#cv2.imshow("output.jpg", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


# dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))
# plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
# plt.show()
# plt.figure()
# dst[0:img.shape[0], 0:img.shape[1]] = img
# cv2.imwrite('output.jpg',dst)
# plt.imshow(dst)
# plt.show()
