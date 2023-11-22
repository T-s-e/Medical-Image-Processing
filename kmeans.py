# from keras.models import load_model
from sklearn.cluster import KMeans
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # 路径设置
    PROJECT_DIR = os.getcwd()
    INPUT = os.path.join(PROJECT_DIR, 'BreastTumor')
    OUTPUT = os.path.join(PROJECT_DIR, 'Result', 'kmeans')

    for file in os.listdir(INPUT):
        img = cv2.imread(os.path.join(INPUT, file), 0)

        cv2.imshow('img', img)

        # 均衡化
        enImg = cv2.medianBlur(cv2.equalizeHist(img), 13)
        enImg = 255 - enImg

        # 增强对比
        alpha = 1.2
        beta = -70
        enImg = cv2.convertScaleAbs(enImg, alpha=alpha, beta=beta)
        # cv2.imshow('enImg', enImg)

        # 中值滤波
        img = cv2.medianBlur(enImg, 13)

        segment = []
        mri1 = img.copy()
        data = np.float32(mri1.reshape((-1, 1)))
        K = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        intensity = np.zeros(K)
        for i in range(K):
            cluster_pixels = data[label == i]
            intensity[i] = np.mean(cluster_pixels)
        idx = np.argmax(intensity)
        label = label.reshape(img.shape)
        mask = np.uint8(label == idx)
        img[mask == 0] = [0]
        # cv2.imshow('kmeans', img)

        # 二值化
        ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow('th', th)

        # 寻找轮廓
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        resultImg = np.zeros_like(th)

        # 初始化最大轮廓和最大面积
        max_contour = None
        max_area = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x > 0 and y > 0 and x + w < th.shape[1] and y + h < th.shape[0]:

                area = cv2.contourArea(contour)

                if area > max_area:
                    max_area = area
                    max_contour = contour

        if max_contour is not None:
            cv2.drawContours(resultImg, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        cv2.imshow('resultImg', resultImg)
        cv2.imwrite(os.path.join(OUTPUT, file), resultImg)

        cv2.waitKey(0)


