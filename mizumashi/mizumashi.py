#!/usr/bin/env python3

# ライブラリのインポート
import cv2
import os

def mizumashi(path, deg, scale):
    """
    水増し画像を生成する関数
    """
    
    image = cv2.imread(path)

    # 画像の回転準備
    target = (int(image.shape[1] / 2), int(image.shape[0] / 2))
    matrix = cv2.getRotationMatrix2D(target, deg, scale)

    # 画像を変換して返す
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

# 水増し画像を保存するフォルダの名前
dirname = "./mizumashi_images"

# 水増し画像を保存するフォルダがなければ作成
if not os.path.isdir(dirname):
    os.mkdir(dirname)

# 水増しする画像
image_path = "./original.jpg"

# 水増し画像を生成(パラメータ: 角度, 拡大率)
print("水増し画像の生成開始")
image1 = mizumashi(image_path, 45, 1.0)
image2 = mizumashi(image_path, 90, 0.7)
image3 = mizumashi(image_path, -45, 1.0)
image4 = mizumashi(image_path, 0, 1.5)

# 水増し画像を保存
print("水増し画像を保存（{}に保存）".format(dirname))
cv2.imwrite(dirname + "/mizumashi1.jpg", image1)
cv2.imwrite(dirname + "/mizumashi2.jpg", image2)
cv2.imwrite(dirname + "/mizumashi3.jpg", image3)
cv2.imwrite(dirname + "/mizumashi4.jpg", image4)
