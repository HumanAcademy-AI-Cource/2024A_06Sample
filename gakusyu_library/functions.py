#!/usr/bin/env python3

# This program is based on the following program.
# https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/functions.py
# Copyright (c) 2016 Koki Saitoh
# The original program is released under the MIT License.
# https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/LICENSE.md

# 必要なライブラリをインポート
import numpy as np


def sigmoid(x):
    """
    シグモイド関数
    """

    return 1 / (1 + np.exp(-x))    

def sigmoid_grad(x):
    """
    シグモイド関数を微分した関数
    """

    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    """
    ソフトマックス関数
    """

    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def cross_entropy_error(y, t):
    """
    交差エントロピー誤差を計算する関数
    """

    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size