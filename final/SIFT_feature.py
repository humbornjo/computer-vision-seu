from motion import *
from utils import *
import cv2,time

def getGoodMatch(des1, des2, ratio=0.6):
    fls=time.time()
    fl = cv2.FlannBasedMatcher()
    fl_matches = fl.knnMatch(des1, des2, k=2)  # des1为模板图，des2为匹配图
    print("FLANN time comsuption: %.4f, # of pairs: %d." % (time.time()-fls,len(fl_matches)))

    #bfs=time.time()
    #bf = cv2.BFMatcher()
    #bf_matches = bf.knnMatch(des1, des2, k=2)  # des1为模板图，des2为匹配图
    #print("BFM time comsuption: %.4f, # of pairs: %d." % (time.time()-bfs,len(bf_matches)))

    matches = sorted(fl_matches, key=lambda x: x[0].distance / x[1].distance)
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
        else:
            break
    if not good:
        raise Exception('No good match under ratio %.2f!' % ratio)
    return good


### Define your parameters here
### Describe the paramters in comments
error_thresh = 3
n_frames = 20

#####################################################
# You may modify the code below to use the params you
# define above.
#####################################################
# Load frames and ground truth bounding boxes
frames = load_frames('Man/img')
gt_bboxes = load_bboxes('Man/groundtruth_rect.txt')

ani = animated_bbox(frames, gt_bboxes)




# Find features to track within the bounding box
x, y, w, h = gt_bboxes[0]
class SiftImg:
    def __init__(self,img):
        self.img= cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    def getFeatures(self,sift):
        self.kp, self.des = sift.detectAndCompute(self.img, None)
sift = cv2.xfeatures2d.SIFT_create()
first=SiftImg(frames[0][y:y + h, x:x + w])
first.getFeatures(sift)
second=SiftImg(frames[1][y:y + h, x:x + w])
second.getFeatures(sift)
good=getGoodMatch(first.des,second.des)
kp=[]
for match in good:
    kp.append(first.kp[match.queryIdx].pt)
keypoints=np.array(kp)

# Shift keypoints by bbox offset
keypoints[:, 1] += x
keypoints[:, 0] += y

# Initailze keypoints and bounding box
kp_I = keypoints
x, y, w, h = gt_bboxes[0]
bboxes = [(x, y, w, h)]

# Start tracking
for i in range(len(frames) - 1):
    I = frames[i]  # Current frame
    J = frames[i + 1]  # Next frame
    flow_vectors = pyramid_lucas_kanade(I, J, kp_I)  # Compute flow vectors
    kp_J = kp_I + flow_vectors  # Estimate keypoints in frame J

    new_keypoints = []
    for yi, xi, yj, xj in np.hstack((kp_I, kp_J)):
        # keypoint falls outside the image
        if yj > J.shape[0] - 2 or yj < 1 or xj > J.shape[1] - 2 or xj < 1:
            print('out of bound')
            continue
        else:
            # Compute error to find lost points
            patch_I = I[int(yi) - 1:int(yi) + 2, int(xi) - 1:int(xi) + 2]
            patch_J = J[int(yj) - 1:int(yj) + 2, int(xj) - 1:int(xj) + 2]
            error = compute_error(patch_I, patch_J)
            if error > error_thresh:
                continue
            else:
                new_keypoints.append([yj, xj])

    # Update keypoints
    kp_I = np.array(new_keypoints)

    # Find bounding box around the keypoints
    if len(kp_I) > 0:
        x = int(kp_I[:, 1].min())
        y = int(kp_I[:, 0].min())
        w = int(kp_I[:, 1].max()) - x
        h = int(kp_I[:, 0].max()) - y
    else:
        (x, y, w, h) = (0, 0, 0, 0)
    bboxes.append((x, y, w, h))

    # Detect new keypoints every n_frames
    if (i + 1) % n_frames == 0 and (w * h > 0):
        I=SiftImg(I[y:y + h, x:x + w])
        I.getFeatures(sift)
        J=SiftImg(J[y:y + h, x:x + w])
        J.getFeatures(sift)
        good = getGoodMatch(I.des, J.des)
        kp = []
        for match in good:
            kp.append(J.kp[match.trainIdx].pt)
        new_keypoints = np.array(kp)
        new_keypoints[:, 1] += x
        new_keypoints[:, 0] += y
        kp_I = np.vstack((kp_I, new_keypoints))

average_iou = 0.0
for gt_bbox, bbox in zip(gt_bboxes, bboxes):
    average_iou += IoU(gt_bbox, bbox)

average_iou /= len(gt_bboxes)
print(average_iou)
