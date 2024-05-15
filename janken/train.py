#!/usr/bin/env python3

# This program is based on the following program.
# https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/trainer.py
# Copyright (c) 2016 Koki Saitoh
# The original program is released under the MIT License.
# https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/LICENSE.md

# 必要なライブラリをインポート
import pickle
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import load_dataset
import os
import sys
sys.path.append(os.path.abspath("../gakusyu_library"))
from network import TwoLayerNetwork
os.environ["QT_LOGGING_RULES"] = "*=false"
matplotlib.rc('font', family='Noto Sans CJK JP')

print("データセットの読み込みを開始します.")
# データセットを読み込む
dataset = load_dataset.load_dataset()
# 訓練画像を取り出す
train_image = dataset['train_image']
# 訓練画像のラベルを取り出す
train_label = dataset['train_label']
# テスト画像を取り出す
test_image = dataset['test_image']
# テスト画像のラベルを取り出す
test_label = dataset['test_label']
print("------------------------------------------------")
print("訓練データ数: {0}個, テストデータ数: {1}個".format(train_image.shape[0], test_image.shape[0]))
print("------------------------------------------------")

# エポック数を定義
epoch = 20
# 訓練画像の数を取得
train_size = train_image.shape[0]
# バッチサイズを定義
batch_size = 10
# 学習率を定義
learning_rate = 0.1
# イテレーション数を算出
iteration_per_epoch = max(train_size / batch_size, 1)
iteration_num = int(iteration_per_epoch * epoch)

# 定義したネットワークをインスタンス化
network = TwoLayerNetwork(3)

# 損失関数の出力結果を保持するリストを定義
train_loss_list = []
# 訓練画像の正解率を保持するリストを定義
train_accuracy_list = []
# テスト画像の正解率を保持するリストを定義
test_accuracy_list = []

print("学習を開始します.")

# 現在のエポック数をカウントする変数を定義
epoch_count = 0

# 時間計測をはじめる
start = time.time()

# 乱数のシード値を指定
np.random.seed(0)

# iteration_num回分、学習を繰り返す
for i in range(iteration_num):
    # バッチサイズ分の訓練画像をランダムに選択
    mask = np.random.choice(train_size, batch_size)
    image_batch = train_image[mask]
    label_batch = train_label[mask]

    # 勾配を計算
    gradient = network.calculate_gradient(image_batch, label_batch)

    # 各層のパラメータ(重みとバイアス)を更新
    for key in ('w_1', 'b_1', 'w_2', 'b_2'):
        network.parameters[key] -= learning_rate * gradient[key]
    
    # 損失を計算
    loss = network.calculate_loss(image_batch, label_batch)
    
    # エポックごとに正解率と損失を保存
    if i % iteration_per_epoch == 0:
        # エポック数をカウント
        epoch_count += 1
        # 訓練データの正解率を計算
        train_accuracy = network.calculate_accuracy(train_image, train_label)
        # テストデータの正解率を計算
        test_accuracy = network.calculate_accuracy(test_image, test_label)
        # 訓練データの正解率をリストに追加
        train_accuracy_list.append(train_accuracy * 100.0)
        # テストデータの正解率をリストに追加
        test_accuracy_list.append(test_accuracy * 100.0)
        # 損失をリストに追加
        train_loss_list.append(loss)
        print("------------------------------------------------")
        print("学習回数: {0}/{1} | 訓練データを使ったときの正解率: {2:5.1f}%, テストデータを使ったときの正解率: {3:5.1f}%".format(epoch_count, epoch, train_accuracy * 100.0, test_accuracy * 100))
        

print("学習が終了しました.")
print("学習にかかった時間：{:.2f}[sec]".format(time.time() - start))

# 学習終了時のパラメータを保存
with open('janken.weights', 'wb') as web:
    pickle.dump(network.parameters, web)
print("学習結果をjanken.weightsに保存します.")


# 精度のグラフを描画
plt.subplot(1, 1, 1)
x = np.arange(len(train_accuracy_list))
plt.plot(x, train_accuracy_list, label='訓練画像の正解率')
plt.plot(x, test_accuracy_list, label='テスト画像の正解率', linestyle='--')
plt.xlabel("-学習回数-")
plt.ylabel("-正解率 [%]-")
plt.ylim(0, 100)
plt.legend(loc='lower right')
plt.show()