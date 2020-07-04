#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import SVNH_DatasetUtil
import pandas as pd
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.models import load_model
import metrics
import time
import numpy as np


# TODO:  mcdc can not provide its useful


def gen_data_mnist(use_adv=False, deepxplore=False, ):
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 28*28
    X_train = X_train.astype('float32').reshape(-1, 28, 28, 1)
    X_test = X_test.astype('float32').reshape(-1, 28, 28, 1)
    X_train /= 255
    X_test /= 255
    model_path = './model/model_mnist.hdf5'
    # model_path = './model/mnist_complicated.h5'
    if use_adv:
        attack_lst = ['fgsm']  # 这里先用一个
        #  attack_lst = ['fgsm', 'jsma', 'bim', 'cw']
        adv_image_all = []
        adv_label_all = []
        for attack in attack_lst:
            adv_image_all.append(np.load('./adv_image/{}_mnist_image.npy'.format(attack)))
            adv_label_all.append(np.load('./adv_image/{}_mnist_label.npy'.format(attack)))
        adv_image_all = np.concatenate(adv_image_all, axis=0)
        adv_label_all = np.concatenate(adv_label_all, axis=0)
        test = np.concatenate([X_test, adv_image_all], axis=0)
        true_test = np.concatenate([Y_test, adv_label_all], axis=0)
    else:
        test = X_test
        true_test = Y_test
    train = X_train
    model = load_model(model_path)
    pred_test_prob = model.predict(test)
    pred_test = np.argmax(pred_test_prob, axis=1)
    input = model.layers[0].output
    if not deepxplore:
        layers = [model.layers[2].output, model.layers[5].output, model.layers[-3].output, model.layers[-2].output,
                  ]

    else:
        layers = [model.layers[1].output, model.layers[3].output, model.layers[4].output, model.layers[8].output,
                  model.layers[8].output, model.layers[9].output, model.layers[10].output]

    layers = list(zip(2 * ['conv'] + 2 * ['dense'], layers))  # 这里要改成对应的层
    # layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))
    print(layers)
    return input, layers, test, train, pred_test, true_test, pred_test_prob, Y_train


def gen_data_cifar(use_adv=True, deepxplore=False):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32*32
    X_train = X_train.astype('float32').reshape(-1, 32, 32, 3)
    X_test = X_test.astype('float32').reshape(-1, 32, 32, 3)
    X_train /= 255
    X_test /= 255
    Y_train = Y_train.reshape(-1)
    Y_test = Y_test.reshape(-1)
    model_path = './model/model_cifar10.h5'
    if use_adv:
        attack_lst = ['cw', 'fgsm', 'jsma', 'bim']
        adv_image_all = []
        adv_label_all = []
        for attack in attack_lst:
            adv_image_all.append(np.load('./adv_image/{}_cifar10_image.npy'.format(attack)))
            adv_label_all.append(np.load('./adv_image/{}_cifar10_label.npy'.format(attack)))
        adv_image_all = np.concatenate(adv_image_all, axis=0)
        adv_label_all = np.concatenate(adv_label_all, axis=0)
        test = np.concatenate([X_test, adv_image_all], axis=0)
        true_test = np.concatenate([Y_test, adv_label_all], axis=0)
    else:
        test = X_test
        true_test = Y_test
    train = X_train
    model = load_model(model_path)
    pred_test_prob = model.predict(test)
    pred_test = np.argmax(pred_test_prob, axis=1)
    input = model.layers[0].output
    lst = []
    for index, layer in enumerate(model.layers):
        if 'activation' in layer.name:
            lst.append(index)
    lst.append(len(model.layers) - 1)
    layers = []
    if not deepxplore:
        for index in lst:
            layers.append(model.layers[index].output)
    else:
        for index in lst:
            if index != len(model.layers) - 1:
                layers.append(model.layers[index - 1].output)
            else:
                layers.append(model.layers[index].output)

    layers = list(zip(19 * ['conv'] + 1 * ['dense'], layers))

    return input, layers, test, train, pred_test, true_test, pred_test_prob, Y_train


def gen_data_fashion(use_adv=True, deepxplore=False):
    path = './fashion-mnist/data/fashion'
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()  ### modify
    X_train = X_train.astype('float32').reshape(-1, 28, 28, 1)
    X_test = X_test.astype('float32').reshape(-1, 28, 28, 1)
    X_train /= 255
    X_test /= 255
    model_path = './model/model_fashion.hdf5'
    if use_adv:
        attack_lst = ['fgsm', 'jsma', 'bim', 'cw']
        adv_image_all = []
        adv_label_all = []
        for attack in attack_lst:
            adv_image_all.append(np.load('./adv_image/{}_fashion_image.npy'.format(attack)))
            adv_label_all.append(np.load('./adv_image/{}_fashion_label.npy'.format(attack)))
        adv_image_all = np.concatenate(adv_image_all, axis=0)
        adv_label_all = np.concatenate(adv_label_all, axis=0)
        test = np.concatenate([X_test, adv_image_all], axis=0)
        true_test = np.concatenate([Y_test, adv_label_all], axis=0)
    else:
        test = X_test
        true_test = Y_test
    train = X_train
    model = load_model(model_path)
    pred_test_prob = model.predict(test)
    pred_test = np.argmax(pred_test_prob, axis=1)
    input = model.layers[0].output
    if not deepxplore:
        layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output, model.layers[6].output,
                  model.layers[8].output, model.layers[9].output, model.layers[10].output]
    else:
        layers = [model.layers[1].output, model.layers[3].output, model.layers[4].output, model.layers[8].output,
                  model.layers[8].output, model.layers[9].output, model.layers[10].output]

    layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))

    return input, layers, test, train, pred_test, true_test, pred_test_prob, Y_train


def gen_data_svhn(use_adv=True, deepxplore=False):
    (X_train, Y_train), (X_test, Y_test) = SVNH_DatasetUtil.load_data()
    Y_train = np.argmax(Y_train, axis=1)  # modify 原本是一堆矩阵,[0,0,0,1,0]代表第四类
    Y_test = np.argmax(Y_test, axis=1)
    model_path = './model/model_svhn.hdf5'
    if use_adv:
        attack_lst = ['cw', 'fgsm', 'jsma', 'bim']
        adv_image_all = []
        adv_label_all = []
        for attack in attack_lst:
            adv_image_all.append(np.load('./adv_image/{}_svhn_image.npy'.format(attack)))
            adv_label_all.append(np.load('./adv_image/{}_svhn_label.npy'.format(attack)))
        adv_image_all = np.concatenate(adv_image_all, axis=0)
        adv_label_all = np.concatenate(adv_label_all, axis=0)
        test = np.concatenate([X_test, adv_image_all], axis=0)
        true_test = np.concatenate([Y_test, adv_label_all], axis=0)
    else:
        test = X_test
        true_test = Y_test
    train = X_train
    model = load_model(model_path)
    pred_test_prob = model.predict(test)
    pred_test = np.argmax(pred_test_prob, axis=1)
    input = model.layers[0].output
    if not deepxplore:
        layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output, model.layers[6].output,
                  model.layers[8].output, model.layers[9].output, model.layers[10].output]
    else:
        layers = [model.layers[1].output, model.layers[3].output, model.layers[4].output, model.layers[8].output,
                  model.layers[6].output, model.layers[9].output, model.layers[10].output]

    layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))

    return input, layers, test, train, pred_test, true_test, pred_test_prob, Y_train


def exp_mcdc(coverage, gen_data, use_adv=False, name="no", d_vc=None, d_dc=None):
    # 初始化变量
    k_bins = 1000
    if use_adv:
        dataset = name + '_adv'
    else:
        dataset = name
    # 计算覆盖
    input, layers, test, train, pred_test, true_test, pred_test_prob, Y_train = gen_data(use_adv,
                                                                                         deepxplore=False)
    # train = train[:5]
    # Y_train = Y_train[:5]
    # print(Y_train)
    # print(pred_test)
    # pred_test = pred_test[:5]
    # true_test = true_test[:5]
    # pred_test_prob = pred_test_prob[:5]
    # test = test[:5]
    model = metrics.MCDC(train, input, layers, d_vc=d_vc, d_dc=d_dc)
    print(layers)
    rate_ssc = model.fit(test)
    print("覆盖率...")
    print(rate_ssc)
    start = time.time()  # 计算排序时间
    # rank_lst2 = model.rank2_ssc()
    end = time.time()

    # 构造结果
    df = pd.DataFrame([])
    # df["LSA"] = score
    #  print(pred_test)

    # df['right'] = (pred_test == true_test).astype('int')  # right
    # df['cam'] = 0  # cam
    # df['ctm'] = 0
    # df['ctm'].loc[rank_lst2] = list(range(1, len(rank_lst2) + 1))  # ctm
    df['rate_ssc'] = rate_ssc  # tate
    # df['rate_svc'] = rate_svc  # tate
    # df['rate_dsc'] = rate_dsc  # tate
    # df['rate_dvc'] = rate_dvc  # tate
    df.to_csv('./all_output/{}_{}.csv'.format(dataset, coverage))
    return start - end


def test():
    dsa = pd.read_csv('./all_output/LSC.csv', index_col=0)
    test_score = dsa["LSA"]
    u = test_score.max()
    print(u)
    k_bins = 1000

    # neuron_mask_activate_test = self.neuron_activate_test[:, self.mask]
    bins = np.linspace(0, u, k_bins)
    x = np.unique(np.digitize(test_score, bins))
    print(len(np.unique(x)) / k_bins)


if __name__ == '__main__':
    dic = {}
    coverage = "MCDC"

    t = exp_mcdc(coverage, gen_data_mnist, name="mnist", use_adv=True)
    # dic['mnist_DSC_k_1000_u_2'] = t

    # t = exp_mcdc(coverage, gen_data_cifar, name="cifar")
    # # dic['cifar_DSC_k_1000_u_2'] = t
    #
    # t = exp_mcdc(coverage, gen_data_fashion, name="fashion")
    # # dic['fashion_DSC_k_1000_u_2'] = t
    #
    # t = exp_mcdc(coverage, gen_data_svhn, name="svhn")
    # # dic['svhn_DSC_k_1000_u_2'] = t
    #
    # t = exp_mcdc(coverage, gen_data_mnist, name="mnist", use_adv=True)
    # # dic['mnist_adv_DSC_k_1000_u_2'] = t
    #
    # t = exp_mcdc(coverage, gen_data_cifar, name="cifar", use_adv=True)
    # # dic['cifar_adv_DSC_k_1000_u_2'] = t
    #
    # t = exp_mcdc(coverage, gen_data_fashion, name="fashion", use_adv=True)
    #
    # t = exp_mcdc(coverage, gen_data_svhn, name="svhn", use_adv=True)
    # # dic['svhn_adv_DSC_k_1000_u_2'] = t

    # res_time_df = pd.DataFrame(dic, index=[0])
    # res_time_df.to_csv("./all_output/" + "output_sa_res_time" + ".csv")
