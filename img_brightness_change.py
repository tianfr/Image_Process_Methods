# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import multiprocessing.dummy as multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
IMG_BIT_DEPTH = 2**8

num_workers = multiprocessing.cpu_count()
# num_workers = 1
print ("CPU number: " + str(num_workers))

# num_workers *= 2


def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]


def func(zip_args):
    img, pic_path, pic = zip_args
    # 把图片转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_img = img
    # 获取灰度图矩阵的行数和列数
    r, c = gray_img.shape[:2]
    dark_sum = 0  # 偏暗的像素 初始化为0个
    dark_prop = 0  # 偏暗像素所占比例初始化为0
    piexs_sum = r*c  # 整个弧度图的像素个数为r*c

    # 遍历灰度图的所有像素
    for row in gray_img:
        for colum in row:
            if colum < 160:  # 人为设置的超参数,表示0~39的灰度值为暗
                dark_sum += 1
    dark_prop = dark_sum/float(piexs_sum)
    print("dark_sum:"+str(dark_sum))
    print("piexs_sum:"+str(piexs_sum))
    print("dark_prop=dark_sum/piexs_sum:"+str(dark_prop))
    if dark_prop >= 0.70:  # 人为设置的超参数:表示若偏暗像素所占比例超过0.78,则这张图被认为整体环境黑暗的图片
        print(pic_path+" is dark!")
        max_num = np.max(img)
        max_time = 255.0/float(max_num)
        max_time_3 = max_time * 3
        max_time_6 = max_time * 6
        max_time_10 = max_time * 10
        res = np.uint8(np.clip((max_time * img + 10), 0, 255))
        res_3 = np.uint8(np.clip((max_time_3 * img + 10), 0, 255))
        res_6 = np.uint8(np.clip((max_time_6 * img + 10), 0, 255))
        res_10 = np.uint8(np.clip((max_time_10 * img + 10), 0, 255))

        tmp = np.hstack((img, res, res_3, res_6, res_10))  # 两张图片横向合并（便于对比显示）
        # for xi in range(0,r):
        #     for xj in range(0,c):
        #         ##set the pixel value increase to 1020%
        #         img[xj,xi,0] = int(img[xj,xi,0]*10) if int(img[xj,xi,0]*10) < IMG_BIT_DEPTH else IMG_BIT_DEPTH
        #         img[xj,xi,1] = int(img[xj,xi,1]*10) if int(img[xj,xi,1]*10) < IMG_BIT_DEPTH else IMG_BIT_DEPTH
        #         img[xj,xi,2] = int(img[xj,xi,2]*10) if int(img[xj,xi,2]*10) < IMG_BIT_DEPTH else IMG_BIT_DEPTH
        #         # img[xj,xi,0] = int(img[xj,xi,0]*10)
        #         # img[xj,xi,1] = int(img[xj,xi,1]*10)
        #         # img[xj,xi,2] = int(img[xj,xi,2]*10)

        cv2.imwrite("../DarkPicDir/"+pic, tmp)  # 把被认为黑暗的图片保存
        # cv2.imshow('12', tmp)
    else:
        print(pic_path+" is bright!")
        # cv2.imshow('12', gray_img)
        # cv2.waitKey(0)
        cv2.imwrite("../DarkPicDir/"+pic, img)
    # hist(pic_path); #若要查看图片的灰度值分布情况,可以这个注释解除

# 用于显示图片的灰度直方图


def hist(pic_path):
    img = cv2.imread(pic_path, 0)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.subplot(121)
    plt.imshow(img, 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("Original")
    plt.subplot(122)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()

# 读取给定目录的所有图片


def readAllPictures(pics_path):
    if not os.path.exists(pics_path):
        print("路径错误，路径不存在！")
        return
    allPics = []
    pics = os.listdir(pics_path)

    pics = list_split(pics, num_workers)

    for pic in tqdm(pics):
        # pic_path = os.path.join(pics_path, pic)
        # allPics.append(pic_path)
        # img = cv2.imread(pic_path)
        # func([img, pic_path, pic])

        pic_path = [os.path.join(pics_path, each) for each in pic]
        allPics.append(pic_path)
        img = [cv2.imread(each) for each in pic_path]
        # Using multiprocessing
        # partial_process_func = partial(func, img=img, pic_path=pic_path)

        zip_args = zip(img, pic_path, pic)

        pool = multiprocessing.Pool(num_workers)

        pool.map_async(func, zip_args)
        # for zip_arg in zip_args:
        #     pool.map_async(func, zip_arg)
        pool.close()
        pool.join()

    return allPics

# 创建用于存放黑暗图片的目录


def createDarkDir():
    DarkDirPath = "../DarkPicDir"
    isExists = os.path.exists(DarkDirPath)
    if not isExists:
        os.makedirs(DarkDirPath)
        print("dark pics dir is created successfully!")
        return True
    else:
        return False


if __name__ == '__main__':
    pics_path = r"D:\Computer Vision\Datasets\prostate_MR\DL_Image"  # 获取所给图片目录
    createDarkDir()
    allPics = readAllPictures(pics_path)
