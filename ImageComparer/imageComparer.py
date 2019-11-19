from skimage import metrics
import numpy as np
#import matplotlib.pyplot as plt
import cv2
import sys
import os
import re

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err

def ssim(imageA, imageB):
    imageA_gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    m = mse(imageA_gray, imageB_gray)
    s = metrics.ssim(imageA_gray, imageB_gray)
    '''
    fig = plt.figure("Image Compare")
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")
 
	# show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")
    
	# show the images
    plt.show()
    '''
    return s

def compareHist(imageA, imageB):
    # Convert it to HSV
    imga_hsv = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)
    imgb_hsv = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
    
    # Calculate the histogram and normalize it
    hist_imga = cv2.calcHist([imga_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_imga, hist_imga, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
    hist_imgb = cv2.calcHist([imgb_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_imgb, hist_imgb, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
    
    # find the metric value
    metric_val = cv2.compareHist(hist_imga, hist_imgb, cv2.HISTCMP_CORREL)
    return metric_val

def main(dir, mins):
    outpath = dir + "../out/"
    # Need to sort images in order by frame number
    images = sorted(os.listdir(dir), key=(lambda p: int(re.search("_[0-9]*", p).group()[1:])))
    
    for ind, image in enumerate(images):
        # print(images[ind])
        ima = None
        if ind == 0:
            path1 = dir + image
            ima = cv2.imread(path1, 1)
            if ima is None:
                raise Exception(f"{path1} not found.")
            cv2.imwrite(outpath + image, ima)
        else:
            ima = imb

        # Write unique first frame to output directory
        
        if ind == len(images) - 1:
            break

        # Check if next frame is similar to current frame    
        path2 = dir + images[ind + 1]
        imb = cv2.imread(path2, 1)
        if imb is None:
            raise Exception(f"{path2} not found.")
        
        s = compareHist(ima, imb)
        print(f"'{image}' and '{images[ind + 1]}': \t s = {s}")
        # If not similar write unique frame to output directory
        if float(s) < mins:
            cv2.imwrite(outpath + images[ind + 1], imb)
        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python imageComparer.py [image-directory] [min simmiliarity required]")
    else:
        main(sys.argv[1], float(sys.argv[2]))