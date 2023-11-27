import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os


class GaborSegmention():
    def __init__(self, img, num_orientations=8):  # 初始化滤波器
        self.img = img
        self.filters = []

        # 定义6个不同尺度和num_orientations个不同方向的Gabor滤波器参数
        ksize = [7, 9, 11, 13, 15, 17]  # 滤波器的大小
        sigma = 4.0  # 高斯函数的标准差
        lambd = 10.0  # 波长
        gamma = 0.5  # 高斯核的椭圆度
        # num_orientations = 8  # 设定多个不同方向的Gabor滤波器

        for theta in np.linspace(0, np.pi, num_orientations, endpoint=False):
            for k in ksize:
                gabor_filter = cv2.getGaborKernel((k, k), sigma, theta, lambd, gamma, ktype=cv2.CV_32F)
                self.filters.append(gabor_filter)

        # 绘制滤波器
        plt.figure(figsize=(12, 12))
        for i in range(len(self.filters)):
            plt.subplot(8, 6, i + 1)
            plt.imshow(self.filters[i])
            plt.axis('off')
        plt.show()

    def getGabor(self):
        feature_matrix = []
        for filter in self.filters:
            # 对图像应用6个不同尺度8个不同方向的Gabor滤波器，得到一个h*w特征图
            filtered_image = cv2.filter2D(self.img, cv2.CV_8UC1, filter)
            # 一个特征图就表示某一个尺度下的某一个方向下的特征
            features = filtered_image.reshape(-1)
            feature_matrix.append(features)

        # 该结果表示每个像素的6个尺度8个方向Gabor特征向量
        feature_matrix = np.array(feature_matrix).T
        return feature_matrix

    def kmeansSeg(self, num_clusters, feature_matrix):
        # 使用Kmeans进行聚类，即计算每个像素的特征向量（48个特征）的相似度
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(feature_matrix)
        # 获取聚类结果
        labels = kmeans.labels_
        return labels

    def colorMap(self, labels):
        # 进行像素映射
        color_map = np.array([[255, 0, 0],  # 蓝色
                              [0, 0, 255],  # 红色
                              [0, 255, 0],  # 绿色
                              [255, 255, 0],
                              [0, 255, 255],
                              [128, 128, 128]
                              ])
        # 将聚类结果转化为图像
        segmented_image = color_map[labels].reshape(self.img.shape[0], self.img.shape[1], 3).astype(np.uint8)
        return segmented_image


if __name__ == "__main__":
    # 加载图像
    image_path = os.path.join('preprocessing', '000018.png')
    img = cv2.imread(image_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(image.shape)

    # 创建gaborSeg分割对象，初始化gabor滤波器
    gaborSeg = GaborSegmention(image)
    # 获取特征矩阵
    feature_matrix = gaborSeg.getGabor()

    # # 分割结果
    # labels=gaborSeg.kmeansSeg(num_clusters=4,feature_matrix=feature_matrix)
    # segmented_image=gaborSeg.colorMap(labels)
    num_clusters = [2, 4, 6]
    seglabels = [gaborSeg.kmeansSeg(num_clusters=num_cluster, feature_matrix=feature_matrix)
                 for num_cluster in num_clusters]

    segmented_images = [gaborSeg.colorMap(labels) for labels in seglabels]

    # 显示结果
    plt.figure(figsize=(16, 8))
    # 原图
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    # 分割图
    for i, segmented_image in enumerate(segmented_images):
        plt.subplot(2, 2, i + 2)
        plt.imshow(segmented_image)
        plt.title("num_clusters={}".format(num_clusters[i]))
        plt.axis('off')

    # plt.subplots_adjust(hspace=0.2)
    plt.show()