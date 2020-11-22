import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from hw02_2018315056_template_matching import template_matching
import time # to be deleted

if __name__ == "__main__":
    # Set the working directory to be the current one
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load a reference image as grayscale
    img_reference = cv2.imread('test_ref4.png', 0)

    # Load a template image as grayscale
    img_template = cv2.imread('fish.png', 0)

    start = time.time()
    # Apply template matching
    x, y, angle, scale = template_matching(img_template, img_reference)
    end = time.time()
    print(f"time taken: {end - start:.4f}s")
    print(f"coordinate:({x}, {y}), angle:{angle}degrees, scale:{scale}")

    h, w = img_template.shape[:2]

    # 画像を傾けてROIを描写。重複部分は後に関数化予定。
    M= cv2.getRotationMatrix2D((w/2, h/2), angle, scale)

    angle_rad = angle/180.0*np.pi
    w_rot = int((np.round(h*np.absolute(np.sin(angle_rad))+w*np.absolute(np.cos(angle_rad))))*scale)
    h_rot = int((np.round(h*np.absolute(np.cos(angle_rad))+w*np.absolute(np.sin(angle_rad))))*scale)
    size_rot = (w_rot, h_rot)

    affine_matrix = M.copy()
    affine_matrix[0][2] = affine_matrix[0][2] -w/2 + w_rot/2
    affine_matrix[1][2] = affine_matrix[1][2] -h/2 + h_rot/2

    rot_img_temp = cv2.warpAffine(img_template, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)

    # _, rot_img_binary = cv2.threshold(rot_img_temp, 127, 255, 0)
    cnt, _ = cv2.findContours(rot_img_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(cnt[0])
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(rot_img_temp, [box], 0, (0, 0, 0), 2)
    img_reference[y:y+h_rot, x:x+w_rot] = rot_img_temp

    # cv2.rectangle(img_reference, (x, y), (x+w_rot, y+h_rot), 0, 2)

    plt.subplot(121),plt.imshow(img_reference,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_reference,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle("TM_CCOEFF_NORMED")

    plt.show()