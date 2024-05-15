#!/usr/bin/env python3

# 必要なライブラリをインポート
import numpy as np
import matplotlib.pyplot as plt
import load_dataset
import os
os.environ["QT_LOGGING_RULES"] = "*=false"

# データセットを読み込む
dataset = load_dataset.load_dataset()
# 訓練画像を取り出す
train_image = dataset['train_image']

# ランダムに訓練画像から20枚を選択
mask = np.random.choice(train_image.shape[0], 20)
select_image = train_image[mask]

# 20枚の画像を表示
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(select_image[i,:].reshape(32,32), cmap='gray')
    plt.axis("off")
plt.show()
