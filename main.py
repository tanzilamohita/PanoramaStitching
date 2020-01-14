# importing libraries
import cv2
import numpy as np

# read input image
img_ = cv2.imread('Image_Set_1/2.jpg')
img_ = cv2.resize(img_, (0, 0), fx=1, fy=1)  # to resize image size i.e. by 50% just change from fx=1 to fx=0.5.
img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
img = cv2.imread('Image_Set_1/1.jpg')
img = cv2.resize(img, (0, 0), fx=1, fy=1)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step1. Keypoints detection using SIFT (Scale Invariant Feature Transform), a very powerful OpenCV algorithm
sift = cv2.xfeatures2d.SIFT_create()

# detect and extract features from the image
def detectAndCompute(image, method=None):
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()

    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)

    return (kps, features)


# find the keypoints and descriptors with SIFT
# kp1 and kp2 are keypoints, des1 and des2 are the descriptors
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
kp1, des1 = detectAndCompute(img1, 'sift')
kp2, des2 = detectAndCompute(img2, 'sift')

print('Keypoints of Image1: ', len(kp1))
print('Keypoints of Image2: ', len(kp2))

cv2.imshow('original_image_1_keypoints', cv2.drawKeypoints(img_, kp1, None))
cv2.imwrite('Image_Set_1/sift_keypoints_1.jpg', cv2.drawKeypoints(img_, kp1, None))
cv2.imshow('original_image_2_keypoints', cv2.drawKeypoints(img, kp2, None))
cv2.imwrite('Image_Set_1/sift_keypoints_2.jpg', cv2.drawKeypoints(img, kp2, None))
#
# Step2. Descriptors matching between two images using FLANNBasedMatcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
match = cv2.FlannBasedMatcher(index_params, search_params)
matches = match.knnMatch(des1, des2, k=2)  # k=2, knnMatcher to give out 2 best matches for each descriptor

# Step2. Descriptors matching between two images using BFMatcher
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m in matches:
    if m[0].distance < 0.5*m[1].distance:
         good.append(m)
    matches = np.asarray(good)

print('There are %d good matches' % (len(good)))
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
img3 = cv2.drawMatchesKnn(img_, kp1, img, kp2, good, None, **draw_params)
cv2.imshow("drawMatches.png", img3)
cv2.imwrite("Image_Set_1/drawMatches.png", img3)


# Step3. Homography estimation using RANSAC
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
cv2.imwrite("Image_Set_1/Warped_Image.jpg", dst)

dst[0:img.shape[0], 0:img.shape[1]] = img
cv2.imshow("output.jpg", trim(dst))
cv2.imwrite("Image_Set_1/output.jpg", trim(dst))
cv2.imshow("output.jpg", dst)
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


