#!/usr/bin/env python3

# This program is based on the following program.
# https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch04/two_layer_net.py
# Copyright (c) 2016 Koki Saitoh
# The original program is released under the MIT License.
# https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/LICENSE.md

# 必要なライブラリをインポート
import numpy as np
from functions import sigmoid, sigmoid_grad, softmax, cross_entropy_error


class TwoLayerNetwork:
    """
    ネットワークの構造定義するクラス
    """

    def __init__(self, labelNum):
        # 乱数のシード値を指定
        np.random.seed(0)
        # 重みを初期化
        self.parameters = {}
        self.parameters['w_1'] = 0.01 * np.random.randn(32*32, 50)
        self.parameters['b_1'] = np.zeros(50)
        self.parameters['w_2'] = 0.01 * np.random.randn(50, labelNum)
        self.parameters['b_2'] = np.zeros(labelNum)

    def load_parameter(self, parameters):
        """
        保存されているパラメータを読み込む関数
        """

        # 保存されているパラメータを代入
        self.parameters['w_1'] = parameters['w_1']
        self.parameters['b_1'] = parameters['b_1']
        self.parameters['w_2'] = parameters['w_2']
        self.parameters['b_2'] = parameters['b_2']

    def predict_label(self, input_image):
        """
        入力画像の数字を推論する関数
        """

        # 計算に使うパラメータを代入
        w_1, w_2 = self.parameters['w_1'], self.parameters['w_2']
        b_1, b_2 = self.parameters['b_1'], self.parameters['b_2']
        
        ## 順伝搬の計算
        # 入力画像を第1層(入力: networkSize, 出力: 50)に入力
        a_1 = np.dot(input_image, w_1) + b_1
        # a_1をシグモイド関数に入力
        z_1 = sigmoid(a_1)
        # z_1を第2層(入力: 50, 出力: 10)に入力
        a_2 = np.dot(z_1, w_2) + b_2
        # a_2をソフトマックス関数に入力
        predict_label = softmax(a_2)
        
        return predict_label
        
    def calculate_loss(self, input_image, correct_label):
        """
        損失を計算する関数
        """

        # 入力画像に対して数字を推論する
        predict_label = self.predict_label(input_image)
        # 推論結果と正解を比較し、誤差を計算する
        loss_value = cross_entropy_error(predict_label, correct_label)

        return loss_value
    
    def calculate_accuracy(self, input_image, correct_label):
        """
        正解率を計算する関数
        """

        # 入力画像に対して数字を推論する
        predict_label = self.predict_label(input_image)
        # one-hot-label形式から数字に変換
        predict_label = np.argmax(predict_label, axis=1)
        correct_label = np.argmax(correct_label, axis=1)
        # バッチで切り取ったすべてのデータの推論結果と正解を比較し、正解率を計算する
        accuracy = np.sum(predict_label == correct_label) / float(input_image.shape[0])
        
        return accuracy

    def calculate_gradient(self, input_image, correct_label):
        """
        勾配を計算する関数
        """
        
        # 計算に使うパラメータを代入
        w_1, w_2 = self.parameters['w_1'], self.parameters['w_2']
        b_1, b_2 = self.parameters['b_1'], self.parameters['b_2']
        
        # 順伝搬の計算
        a_1 = np.dot(input_image, w_1) + b_1
        z_1 = sigmoid(a_1)
        a_2 = np.dot(z_1, w_2) + b_2
        predict_label = softmax(a_2)
        
        # 逆伝搬の計算
        gradients = {}

        # 第2層の勾配を計算
        dy = (predict_label - correct_label) / input_image.shape[0]
        gradients['w_2'] = np.dot(z_1.T, dy)
        gradients['b_2'] = np.sum(dy, axis=0)
        
        # 第1層の勾配を計算
        dz_1 = np.dot(dy, w_2.T)
        da_1 = sigmoid_grad(a_1) * dz_1
        gradients['w_1'] = np.dot(input_image.T, da_1)
        gradients['b_1'] = np.sum(da_1, axis=0)

        return gradients