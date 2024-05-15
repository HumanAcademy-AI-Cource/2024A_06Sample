#!/usr/bin/env python3

# 必要なライブラリをインポート
import numpy as np
import cv2
import os

dirnames = ["gu", "choki", "pa"]
dataset_name = "janken_dataset"

def load_dataset():
    """
    データセットを生成する関数
    """

    dataset = {}
    # 訓練用データセットの読み込み
    train_color_image = []
    train_image = []
    train_label = []
    for dirname in dirnames:
        files = os.listdir("{}/train_dataset/{}/".format(dataset_name, dirname))
        for file in files:
            image = cv2.imread("{}/train_dataset/{}/{}".format(dataset_name, dirname, file))
            train_color_image.append(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.flatten() # 一次元化
            image = 255 - image # 0-255を反転
            image = image.astype(np.float32) #型を変更（後で割り算をするため）
            image /= 255.0 # 正規化
            label = np.zeros(3)
            if dirname == "gu":
                index = 0
            if dirname == "choki":
                index = 1
            if dirname == "pa":
                index = 2
            label[index] = 1.0
            train_image.append(image)
            train_label.append(label)
    dataset['train_color_image'] = np.array(train_color_image)
    dataset['train_image'] = np.array(train_image)
    dataset['train_label'] = np.array(train_label)

    # テスト用データセットの読み込み
    test_color_image = []
    test_image = []
    test_label = []
    for dirname in dirnames:
        files = os.listdir("{}/test_dataset/{}/".format(dataset_name, dirname))
        for file in files:
            image = cv2.imread("{}/test_dataset/{}/{}".format(dataset_name, dirname, file))
            test_color_image.append(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.flatten() # 一次元化
            image = 255 - image # 0-255を反転
            image = image.astype(np.float32) #型を変更（後で割り算をするため）
            image /= 255.0 # 正規化
            label = np.zeros(3)
            if dirname == "gu":
                index = 0
            if dirname == "choki":
                index = 1
            if dirname == "pa":
                index = 2
            label[index] = 1.0
            test_image.append(image)
            test_label.append(label)
    dataset['test_color_image'] = np.array(test_color_image)
    dataset['test_image'] = np.array(test_image)
    dataset['test_label'] = np.array(test_label)

    return dataset