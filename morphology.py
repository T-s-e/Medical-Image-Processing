import numpy as np
import cv2
import os

# 路径设置
PROJECT_DIR = os.getcwd()
INPUT = os.path.join(PROJECT_DIR, 'BreastTumor')
OUTPUT = os.path.join(PROJECT_DIR, 'Result', 'morphology')


if __name__ == '__main__':

    # 文件载入
    for file in os.listdir(INPUT):
        print(file)
        img = cv2.imread(os.path.join(INPUT, file), 0)
        cv2.imshow('img', img)

        # 均衡化
        enImg = cv2.medianBlur(cv2.equalizeHist(img), 13)
        enImg = 255-enImg

        # 增强对比
        alpha2 = 1.3
        beta2 = -100
        enImg = cv2.convertScaleAbs(enImg, alpha=alpha2, beta=beta2)

        # 中值滤波
        blur = cv2.medianBlur(enImg, 7)

        # 二值化
        ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 闭运算
        kernel = np.ones((9,9), np.uint8)
        closeImg = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        cv2.imshow('closeImg', closeImg)

        # 寻找轮廓
        contours, _ = cv2.findContours(closeImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        resultImg = np.zeros_like(closeImg)

        # 初始化最大轮廓和最大面积
        max_contour = None
        max_area = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x > 0 and y > 0 and x + w < closeImg.shape[1] and y + h < closeImg.shape[0]:

                area = cv2.contourArea(contour)

                if area > max_area:
                    max_area = area
                    max_contour = contour

        if max_contour is not None:
            cv2.drawContours(resultImg, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        # 图像保存
        cv2.imwrite(os.path.join(OUTPUT, file), resultImg)


        # 结果
        cv2.imshow('resultImg', resultImg)
        cv2.waitKey(0)

