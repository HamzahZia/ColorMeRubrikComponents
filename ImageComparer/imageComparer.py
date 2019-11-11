from skimage import metrics
import numpy as np
import matplotlib.pyplot as plt
import cv2

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	return err

def compare_image(imageA, imageB):
    imageA_gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    m = mse(imageA_gray, imageB_gray)
    s = metrics.structural_similarity(imageA_gray, imageB_gray)

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

    return s

ima = cv2.imread("images/suits3.png", 1)
imb = cv2.imread("images/suits4.png", 1)
compare_image(ima, imb)
