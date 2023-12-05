import os
import cv2

def img_size(img):
    white = 0
    height = img.shape[0]
    width = img.shape[1]
    for row in range(height):
        for col in range(width):
            val = img[row][col]
            if val == 255:
                white = white + 1
    return white

def size_same(resultImg,img_annotation):
    size = 0
    height = resultImg.shape[0]
    width = resultImg.shape[1]
    # print(resultImg.shape[1])
    # print(img_annotation.shape[1])
    for row in range(height):
        for col in range(width):
            val1 = resultImg[row][col]
            val2 = img_annotation[row][col]
            if val1 == 255 & val2 == 255:
                size = size + 1
    return size

def find_TF(resultImg,img_annotation):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    height = resultImg.shape[0]
    width = resultImg.shape[1]
    for row in range(height):
        for col in range(width):
            val1 = resultImg[row][col]
            val2 = img_annotation[row][col]
            if val2 == 255:
                if val1 == 255:
                    TP = TP + 1
                else:
                    FP = FP + 1
            else:
                if val1 == 255:
                    FN = FN + 1
                else:
                    TN = TN + 1
    return(TP,FP,FN,TN)

if __name__ == '__main__':
    PROJECT_DIR = os.getcwd()
    # OUTPUT = os.path.join(PROJECT_DIR, 'Result', 'kmeans')
    # OUTPUT = os.path.join(PROJECT_DIR, 'Result', 'morphology')
    # OUTPUT = os.path.join(PROJECT_DIR, 'Result', 'GraphCut_Intensity')
    # OUTPUT = os.path.join(PROJECT_DIR, 'Result', 'UNet')
    # OUTPUT = os.path.join(PROJECT_DIR, 'Result', 'Gabor')
    # OUTPUT = os.path.join(PROJECT_DIR, 'Result', 'NCut')
    OUTPUT = os.path.join(PROJECT_DIR, 'Result', 'activeContour')
    annotation = os.path.join(PROJECT_DIR,'Annotation')

    for file in os.listdir(OUTPUT):
        print(file)
        resultImg = cv2.imread(os.path.join(OUTPUT, file), 0)
        for file2 in os.listdir(annotation):
            if file2 == file:
                break
        img_annotation = cv2.imread(os.path.join(annotation, file2),0)

        white = img_size(resultImg)
        white_ann = img_size(img_annotation)
        size = size_same(resultImg,img_annotation)
        dice = 2*size/(white+white_ann)
        print('Dice:',dice)

        # 计算Accuracy、Specifity、Sensitivity
        TP,FP,FN,TN = find_TF(resultImg,img_annotation)
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        specificity = TN/(TN+FP)
        sensitivity = TP/(TP+FN)
        print('Sensitivity:',sensitivity)
        print('Specificity',specificity)
        print('Accuracy',accuracy)