import cv2 as cv2
import matplotlib.cm as cm
import numpy as np
import pylab as plb
import copy
from scipy.spatial.distance import cdist
import os


# 路径设置
PROJECT_DIR = os.getcwd()
INPUT = os.path.join(PROJECT_DIR, '..', 'BreastTumor')
OUTPUT = os.path.join(PROJECT_DIR, '..', 'Result', 'activeContour')

_ALPHA = 300
_BETA = 2
_W_LINE = 80
_W_EDGE = 80
_NUM_NEIGHBORS = 9

# the candaidate 8 movment
neighbors = np.array([[i, j] for i in range(-1, 2) for j in range(-1, 2)])


# internal forces = summation from i to n alph * |Vi+1 - Vi|2 + BETA * |Vi+1 - 2* Vi + Vi-1 |2
def internalEnergy(snake):
    iEnergy = 0
    snakeLength = len(snake)
    for index in range(snakeLength - 1, -1, -1):
        nextPoint = (index + 1) % snakeLength
        currentPoint = index % snakeLength
        previousePoint = (index - 1) % snakeLength
        iEnergy = iEnergy + (_ALPHA * (np.linalg.norm(snake[nextPoint] - snake[currentPoint]) ** 2)) \
                  + (_BETA * (np.linalg.norm(snake[nextPoint] - 2 * snake[currentPoint] + snake[previousePoint]) ** 2))
    return iEnergy


# total energy for internal and external without constant
def totalEnergy(grediant, image, snake):
    iEnergy = internalEnergy(snake)
    eEnergy = externalEnergy(grediant, image, snake)
    tEnergy = iEnergy + eEnergy

    return tEnergy


# plot
def _display(image, changedPoint=None, snaxels=None):
    plb.clf()
    if snaxels is not None:
        for s in snaxels:
            if (changedPoint is not None and (s[0] == changedPoint[0] and s[1] == changedPoint[1])):
                plb.plot(s[0], s[1], 'r', markersize=10.0)
            else:
                plb.plot(s[0], s[1], 'g.', markersize=10.0)

    plb.imshow(image, cmap=cm.Greys_r)
    plb.draw()

    return


# external forces summation of image grediant for all contour points
def externalEnergy(grediant, image, snak):
    sum = 0
    snaxels_Len = len(snak)
    for index in range(snaxels_Len - 1):
        point = snak[index]
        # 添加检查，确保 point 的坐标在图像范围内
        if 0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0]:
            sum = sum + image[point[1], point[0]]
    pixel = 255 * sum

    eEnergy = _W_LINE * pixel - _W_EDGE * imageGradient(grediant, snak)

    return eEnergy



def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)


def basicImageGradiant(image):
    s_mask = 17
    sobelx = np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=s_mask))
    sobelx = interval_mapping(sobelx, np.min(sobelx), np.max(sobelx), 0, 255)
    sobely = np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=s_mask))
    sobely = interval_mapping(sobely, np.min(sobely), np.max(sobely), 0, 255)
    gradient = 0.5 * sobelx + 0.5 * sobely
    return gradient


def imageGradient(gradient, snak):
    sum = 0
    snaxels_Len = len(snak)
    for index in range(snaxels_Len - 1):
        point = snak[index]
        # 添加检查，确保 point 的坐标在图像范围内
        if 0 <= point[0] < gradient.shape[1] and 0 <= point[1] < gradient.shape[0]:
            sum = sum + gradient[point[1], point[0]]
    return sum



def isPointInsideImage(image, point):
    return np.all(point < np.shape(image)) and np.all(point > 0)


def _pointsOnCircle(center, radius, num_points=12):
    points = np.zeros((num_points, 2), dtype=np.int32)
    for i in range(num_points):
        theta = float(i) / num_points * (2 * np.pi)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        p = [x, y]
        points[i] = p

    return points


def postprocess_contour(snake):
    # 计算每个点之间的距离矩阵
    distances = cdist(snake, snake)

    # 找到每个点的最近邻
    nearest_neighbors = np.argmin(distances, axis=1)

    # 生成连续轮廓
    new_contour = snake[nearest_neighbors]

    return new_contour


def activeContour(image_file, center, radius):
    image = cv2.imread(os.path.join(INPUT, image_file), 0)
    plb.ion()
    plb.figure(figsize=np.array(np.shape(image)) / 50.)

    snake = _pointsOnCircle(center, radius, 50)
    grediant = basicImageGradiant(image)
    plb.ioff()

    snakeColon = copy.deepcopy(snake)
    indexOFlessEnergy = 0  # 初始化 indexOFlessEnergy

    for i in range(100):
        for index, point in enumerate(snake):
            min_energy2 = float("inf")
            for cindex, movement in enumerate(neighbors):
                next_node = (point + movement)
                if not isPointInsideImage(image, next_node):
                    continue
                if not isPointInsideImage(image, point):
                    continue

                snakeColon[index] = next_node

                totalEnergyNext = totalEnergy(grediant, image, snakeColon)

                if (totalEnergyNext < min_energy2):
                    min_energy2 = copy.deepcopy(totalEnergyNext)
                    indexOFlessEnergy = copy.deepcopy(cindex)
            snake[index] = (snake[index] + neighbors[indexOFlessEnergy])
        snakeColon = copy.deepcopy(snake)

    # plb.ioff()
    # _display(image, None, snake)
    # plb.plot()
    # # plb.savefig(os.path.splitext(image_file)[0] + "-segmented.png")
    # plb.show()

    plb.ioff()
    _display(image, None, snake)
    plb.plot()

    # 后处理，获取连续轮廓
    new_contour = postprocess_contour(snake)

    # 显示连续轮廓
    plb.plot(new_contour[:, 0], new_contour[:, 1], 'b-', markersize=10.0)
    plb.show()

    # 创建空白图像
    binary_image = np.zeros_like(image)

    # 使用连续轮廓填充图像
    cv2.fillPoly(binary_image, [new_contour.astype(int)], 255)

    # 显示填充后的图像
    cv2.imshow('Filled Image', binary_image)
    cv2.imwrite(os.path.join(OUTPUT, image_file), binary_image)
    cv2.waitKey(0)

    return


if __name__ == '__main__':
    files = os.listdir(INPUT)
    num = len(files)

    centers = [(320, 103), (231, 129), (210, 162), (211, 101), (272, 171), (149, 121), (149, 121), (263, 147), (276, 228), (318, 146)]
    radius = [40, 60, 141, 88, 160, 44, 44, 100, 122, 82]

    for i in range(num):
        print('processing: ', files[i])
        activeContour(files[i], centers[i], radius[i])

        # img = cv2.imread(os.path.join(INPUT, files[i]), 0)
        # plb.ioff()
        # _display(img)
        # plb.plot()

        plb.show()
        # exit()



