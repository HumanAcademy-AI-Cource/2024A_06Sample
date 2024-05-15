#!/usr/bin/env python3

from PIL import Image, ImageFilter, ImageEnhance
import os
import shutil
import numpy as np


dataset_name = "tegaki_dataset"
dirnames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
original_dirname = "original_images"
train_dirname = "train_dataset"
test_dirname = "test_dataset"

train_choice = ["001", "002", "003", "005", "007", "008", "009", "010"]
test_choice = ["004", "006"]

# 水増しする枚数を指定
mizumashi_num = 100

# 乱数のシード値を指定
np.random.seed(0)

def resize_train_mizumashi(img):
    rotated_img = img.convert('RGBA').rotate(np.random.randint(-20, 20))
    kasamashi_img = Image.new('RGBA', rotated_img.size, (255,) * 4)
    kasamashi_img = Image.composite(rotated_img, kasamashi_img, rotated_img)
    kasamashi_img = kasamashi_img.convert(img.mode)
    kasamashi_img = kasamashi_img.resize((32, 32))
    kasamashi_img = ImageEnhance.Color(kasamashi_img).enhance(np.random.uniform(0.5, 1.5))
    kasamashi_img = kasamashi_img.filter(ImageFilter.GaussianBlur(np.random.randint(0, 4)))
    return kasamashi_img

def resize_test_mizumashi(img):
    rotated_img = img.convert('RGBA').rotate(np.random.randint(-20, 20))
    kasamashi_img = Image.new('RGBA', rotated_img.size, (255,) * 4)
    kasamashi_img = Image.composite(rotated_img, kasamashi_img, rotated_img)
    kasamashi_img = kasamashi_img.convert(img.mode)
    kasamashi_img = kasamashi_img.resize((32, 32))
    kasamashi_img = ImageEnhance.Color(kasamashi_img).enhance(np.random.uniform(0.5, 1.5))
    return kasamashi_img

# ディレクトリ作成
if(os.path.exists(dataset_name)):
    shutil.rmtree(dataset_name)
os.makedirs("{}".format(dataset_name))

# 訓練用画像
print("訓練用データセットを作成中")
os.makedirs("{}/{}/".format(dataset_name, train_dirname))
for dirname in dirnames:
    print("  「{}」ディレクトリからデータ作成中".format(dirname))
    files = os.listdir("{}/{}/".format(original_dirname, dirname))
    os.makedirs("{}/{}/{}".format(dataset_name, train_dirname, dirname))
    for file in files:
        img = Image.open("{}/{}/{}".format(original_dirname, dirname, file))
        if(os.path.splitext(os.path.basename(file))[0] in train_choice):
            for i in range(mizumashi_num):
                kasamashi_img = resize_train_mizumashi(img)
                kasamashi_img.save("{}/{}/{}/M{}_{}".format(dataset_name, train_dirname, dirname, str(i), file))

# テスト用画像
print("テスト用データセットを作成中")
os.makedirs("{}/{}/".format(dataset_name, test_dirname))
for dirname in dirnames:
    print("  「{}」ディレクトリからデータ作成中".format(dirname))
    files = os.listdir("{}/{}/".format(original_dirname, dirname))
    os.makedirs("{}/{}/{}".format(dataset_name, test_dirname, dirname))
    for file in files:
        img = Image.open("{}/{}/{}".format(original_dirname, dirname, file))
        if(os.path.splitext(os.path.basename(file))[0] in test_choice):
            for i in range(mizumashi_num):
                kasamashi_img = resize_test_mizumashi(img)
                kasamashi_img.save("{}/{}/{}/M{}_{}".format(dataset_name, test_dirname, dirname, str(i), file))
print("データセットの作成が完了しました。")