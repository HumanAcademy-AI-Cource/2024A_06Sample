#!/usr/bin/env python3

# 必要なライブラリをインポート
import sys
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import load_dataset
import os
import sys
sys.path.append(os.path.abspath("../gakusyu_library"))
from network import TwoLayerNetwork
os.environ["QT_LOGGING_RULES"] = "*=false"


# データセットを読み込む
dataset = load_dataset.load_dataset()
# テスト画像を取り出す
test_image = dataset['test_image']
# テスト画像（カラー）を取り出す
test_color_image = dataset['test_color_image']
# テスト画像のラベルを取り出す
test_label = dataset['test_label']

# 定義したネットワークをインスタンス化
network = TwoLayerNetwork(3)

# 保存されたパラメータを読み込む
with open('janken.weights', 'rb') as web:
    params = pickle.load(web)
# パラメータをモデルに適用する
network.load_parameter(params)

# 乱数のシード値設定
np.random.seed()

# ランダムにテストデータを1枚を選択
mask = np.random.choice(test_image.shape[0], 1)
image_batch = test_image[mask]
image_color_batch = test_color_image[mask]
label_batch = test_label[mask]

names = ["グー", "チョキ", "パー"]

def key_press(event):
    """
    キー入力されたとき、予測を開始する関数
    """

    # wキーを押したとき、予測を開始する
    if event.key == "w":
        # 選択したテストデータに対して予測開始
        predict_label = network.predict_label(image_batch)
        print("---------------------")
        for i in range(3):
            print("{0}:　{1:.0f}%".format(names[i], predict_label[0][i] * 100))
        print("---------------------")
        print("予測結果: {0}".format(names[np.argmax(predict_label, axis=1)[0]]))
        print("入力画像の正解ラベル: {0}".format(names[np.argmax(label_batch, axis=1)[0]]))


# 入力画像を表示
print("入力画像を表示します.")
print("キーを入力してください.")
print("w: 予測開始, q: 終了")
plt.imshow(cv2.cvtColor(image_color_batch.reshape(32,32,3), cv2.COLOR_BGR2RGB))
plt.xticks(color="None")
plt.yticks(color="None")
plt.tick_params(length=0)
plt.connect('key_press_event',key_press)
plt.show()