#!usr/bin/env python3
# -*- coding:utf-8 -*-
# --------------------------
"""
@author:Sui yue
@describe: 对比增强 伽马变换
@time: 2019/09/18 22:22:51
"""
import cv2
import numpy as np
import sys
# 主函数
if __name__ == '__main__':
    I = cv2.imread(
        'D:\Computer Vision\Datasets\prostate_MR\DL_Image\DL_Image0001_0001.png', cv2.IMREAD_GRAYSCALE)
    # 图像归一化
    fI = I/255.0
    # 伽马变换
    gamma = 0.3
    O = np.power(fI, gamma)
    # 显示原图和伽马变换
    cv2.imshow("I", I)
    cv2.imshow("O", O)
    cv2.waitKey()
    cv2.destroyAllWindows()
