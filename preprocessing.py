import cv2
import os


if __name__ == '__main__':
    files = os.listdir('BreastTumor')
    for file in files:
        img = cv2.imread(os.path.join('BreastTumor', file), 0)

        # 均衡化
        enImg = cv2.medianBlur(cv2.equalizeHist(img), 13)
        enImg = 255 - enImg

        # 增强对比
        alpha2 = 1.3
        beta2 = -100
        enImg = cv2.convertScaleAbs(enImg, alpha=alpha2, beta=beta2)

        # 中值滤波
        blur = cv2.medianBlur(enImg, 7)
        # res = cv2.resize(blur, (0, 0), fx=0.2, fy=0.2)
        cv2.imwrite(os.path.join('preprocessing', file), blur)


