from skimage import metrics
import numpy as np
import cv2
import sys
import os
import re
import time
# import matplotlib.pyplot as plt

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
    cv2.normalize(hist_imga, hist_imga, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_imgb = cv2.calcHist([imgb_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_imgb, hist_imgb, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # find the metric value
    metric_val = cv2.compareHist(hist_imga, hist_imgb, cv2.HISTCMP_CORREL)
    return metric_val

def main(dir, minr):
    # Output directory for frames to keep
    outpath = dir + "../out/"
    # Need to sort images in order by frame number
    og_images = sorted(os.listdir(dir), key=(lambda p: int(re.search("_[0-9]+", p).group()[1:])))
    print(f"Number of frames: {len(og_images)}")
    ''' 
     compareNeighbours: compares each image in images to its immediate successor in the array
                        if they are too similar it will discard the successor
                        else it will add the successor to the returned array 
    '''
    def compareNeighbours(images):
        ret = []
        for ind, image in enumerate(images):
            if ind == len(images) - 1:
                break

            ima = None
            if ind == 0:
                path1 = dir + image
                ima = cv2.imread(path1, 1)
                if ima is None:
                    raise Exception(f"{path1} not found.")
                # first frame is unique and should be added to return array
                ret.append(image)
            else:
                ima = imb

            # Check if next frame is similar to current frame    
            path2 = dir + images[ind + 1]
            imb = cv2.imread(path2, 1)
            if imb is None:
                raise Exception(f"{path2} not found.")
            
            r = compareHist(ima, imb)
            #print(f"'{image}' and '{images[ind + 1]}': \t r = {r}")

            # If not similar add unique frame to return array
            if float(r) < minr:
                ret.append(images[ind + 1])
        return ret
    
    discarded_frames = 0
    iterations = 0
    while True:
        iterations += 1
        filtered_images = compareNeighbours(og_images)
        '''
        if length of output array is equal to input array no images were filtered
            thus all frames are *unique*
        '''
        if len(filtered_images) == len(og_images):
            print(f"Discarded frames: {discarded_frames}")
            print(f"Iterations: {iterations}")
            for im_name in filtered_images:
                image = cv2.imread(dir + im_name)
                cv2.imwrite(outpath + im_name, image)
            return
        else:
            discarded_frames += len(og_images) - len(filtered_images)
            og_images = filtered_images
    
        
if __name__ == "__main__":
    start = time.time()
    if len(sys.argv) != 3:
        print("Usage: python imageComparer.py [image-directory] [min similarity required]")
    else:
        main(sys.argv[1], float(sys.argv[2]))
    print(f"Execution time: {time.time() - start} seconds")