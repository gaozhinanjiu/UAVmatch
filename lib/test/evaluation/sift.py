from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
def Splice(img1, img2):

    try:
        #img1_Gauss = cv2.GaussianBlur(img1, (3, 3), 0)

        #img2_Gauss = cv2.GaussianBlur(img2, (3, 3), 0)
        # img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img1_Gauss = cv2.GaussianBlur(img1, (3, 3), 0)

        img2_Gauss = cv2.GaussianBlur(img2, (3, 3), 0)

        # plt.imshow(img1_Gauss), plt.show()
        # plt.imshow(img2_Gauss), plt.show()



    # # 实例化SIFT
        sift =cv2.ORB_create(10000)


    # 得到两幅图像的特征点
        kp1, des1 = sift.detectAndCompute(img1_Gauss, None)
        kp2, des2 = sift.detectAndCompute(img2_Gauss, None)

    # plt.imshow(img_KeypointsDraw), plt.show()
    # cv2.imwrite('./Pics/KeyPoints.png',img_KeypointsDraw)

    # 实例化匹配器
        bf = cv2.BFMatcher()
    # 匹配特征点，采用1NN（1近邻）匹配
        matches = bf.knnMatch(des1, des2, k=1)

    # 画出并保存匹配结果(仅在实验中使用)
    # img_MatchesDraw = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
    # plt.imshow(img_MatchesDraw), plt.show()
    # cv2.imwrite('./Pics/Matches1NN.png',img_MatchesDraw)

    # 重新匹配特征点，并采用1NN/2NN<0.7的方式筛选出好的匹配对
        matches = bf.knnMatch(des1, des2, k=2)

    # good_matches用于保存好的匹配对
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)
        if len(good_matches) >= 3:
            flag=True
    # 画出并保存匹配结果（仅在实验中使用）
    # img_MatchesDraw = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
    # plt.imshow(img_MatchesDraw), plt.show()
    # cv2.imwrite('./Pics/Matches2NN.png',img_MatchesDraw)

    # 得到良好匹配对的坐标对
            good_kp1 = []
            good_kp2 = []
            for m in good_matches:
                good_kp1.append(kp1[m.queryIdx].pt)
                good_kp2.append(kp2[m.trainIdx].pt)

            good_kp1 = np.array(good_kp1)
            good_kp2 = np.array(good_kp2)
    # 用RANSAC算法得到最佳透视变换矩阵
    # RANSAC参数
            ransac_reproj_threshold = 2.0  # RANSAC重投影阈值
            ransac_max_iter = 100  # RANSAC最大迭代次数

    # 使用RANSAC估计仿射变换矩阵
            affine_matrix, _ = cv2.estimateAffine2D(good_kp1, good_kp2,
                                            ransacReprojThreshold=ransac_reproj_threshold,
                                            maxIters=ransac_max_iter)
            affine_matrix[0 ,2] = affine_matrix[0 ,2] / 480
            affine_matrix[1, 2] = affine_matrix[1, 2] / 480

            affine_matrix=torch.from_numpy(affine_matrix).type(torch.float32)
            affine_matrix = affine_matrix.view(6,1)
    #retval, mask = cv2.findHomography(good_kp2, good_kp1, cv2.RANSAC, confidence=0.997)


            return affine_matrix,flag
        else:
            flag=False
            affine_matrix=None
            return affine_matrix,flag
    except Exception as e:
        flag = False
        affine_matrix = None
        return affine_matrix, flag

