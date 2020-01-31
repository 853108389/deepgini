#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# TODO: 重命名为sa_exp,同时将执行脚本exc_main.py内容也修改为sa_exp
import SVNH_DatasetUtil
import pandas as pd
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.models import load_model
import metrics
import time
import numpy as np

import model_conf


def gen_data_mnist(model_name, use_adv=False, deepxplore=False, ):
    model_path = model_conf.get_model_path(model_conf.mnist, model_name)  #
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 28*28
    X_train = X_train.astype('float32').reshape(-1, 28, 28, 1)
    X_test = X_test.astype('float32').reshape(-1, 28, 28, 1)
    X_train /= 255
    X_test /= 255
    # model_path = './model/model_mnist.hdf5'
    if use_adv:
        attack_lst = ['fgsm', 'jsma', 'bim', 'cw']
        adv_image_all = []
        adv_label_all = []
        for attack in attack_lst:
            im, lab = model_conf.get_adv_path(attack, model_conf.mnist, model_name)
            adv_image_all.append(np.load(im))
            adv_label_all.append(np.load(lab))
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
    if not deepxplore:  #
        if model_name == model_conf.LeNet5:
            layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output, model.layers[6].output,
                      model.layers[8].output, model.layers[9].output, model.layers[10].output]
            layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))  # 激活,池化,全连接
        else:
            layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output, model.layers[6].output,
                      model.layers[8].output, ]
            # print(model.layers[1], model.layers[3], model.layers[6])
            layers = list(zip(4 * ['conv'] + 1 * ['dense'], layers))  #
    else:
        if model_name == model_conf.LeNet5:
            layers = [model.layers[1].output, model.layers[3].output, model.layers[4].output, model.layers[8].output,
                      model.layers[8].output, model.layers[9].output, model.layers[10].output]
            layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))  # 卷积,池化,全连接
        else:
            layers = [model.layers[1].output, model.layers[3].output, model.layers[4].output, model.layers[6].output,
                      model.layers[8].output, ]
            layers = list(zip(4 * ['conv'] + 1 * ['dense'], layers))

    return input, layers, test, train, pred_test, true_test, pred_test_prob, Y_train


def gen_data_cifar(model_name, use_adv=True, deepxplore=False):
    model_path = model_conf.get_model_path(model_conf.cifar10, model_name)
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32*32
    X_train = X_train.astype('float32').reshape(-1, 32, 32, 3)
    X_test = X_test.astype('float32').reshape(-1, 32, 32, 3)
    X_train /= 255
    X_test /= 255
    Y_train = Y_train.reshape(-1)
    Y_test = Y_test.reshape(-1)
    # model_path = './model/model_cifar10.h5'
    if use_adv:
        attack_lst = ['cw', 'fgsm', 'jsma', 'bim']
        adv_image_all = []
        adv_label_all = []
        for attack in attack_lst:
            im, lab = model_conf.get_adv_path(attack, model_conf.cifar10, model_name)
            adv_image_all.append(np.load(im))
            adv_label_all.append(np.load(lab))
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
    if not deepxplore:
        if model_name == model_conf.resNet20:
            layers = []
            for index in lst:
                layers.append(model.layers[index].output)
            layers = list(zip(19 * ['conv'] + 1 * ['dense'], layers))
        else:  # Vgg16
            layers = []
            for i in range(1, 19):
                layers.append(model.layers[i].output)
            for i in range(20, 23):
                layers.append(model.layers[i].output)
            layers = list(zip(18 * ['conv'] + 3 * ['dense'], layers))
    else:
        if model_name == model_conf.resNet20:
            layers = []
            for index in lst:
                if index != len(model.layers) - 1:
                    layers.append(model.layers[index - 1].output)
                else:
                    layers.append(model.layers[index].output)
            layers = list(zip(19 * ['conv'] + 1 * ['dense'], layers))
        else:
            layers = []
            for i in range(1, 19):
                layers.append(model.layers[i].output)
            for i in range(20, 23):
                layers.append(model.layers[i].output)
            layers = list(zip(18 * ['conv'] + 3 * ['dense'], layers))

    return input, layers, test, train, pred_test, true_test, pred_test_prob, Y_train


def gen_data_fashion(model_name, use_adv=True, deepxplore=False):
    model_path = model_conf.get_model_path(model_conf.fashion, model_name)
    path = './fashion-mnist/data/fashion'
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()  ### modify
    X_train = X_train.astype('float32').reshape(-1, 28, 28, 1)
    X_test = X_test.astype('float32').reshape(-1, 28, 28, 1)
    X_train /= 255
    X_test /= 255
    # model_path = './model/model_fashion.hdf5'
    if use_adv:
        attack_lst = ['fgsm', 'jsma', 'bim', 'cw']
        adv_image_all = []
        adv_label_all = []
        for attack in attack_lst:
            im, lab = model_conf.get_adv_path(attack, model_conf.fashion, model_name)
            adv_image_all.append(np.load(im))
            adv_label_all.append(np.load(lab))
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
    if not deepxplore:
        if model_name == model_conf.LeNet5:  # 选择模型
            layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output, model.layers[6].output,
                      model.layers[8].output, model.layers[9].output, model.layers[10].output]
            layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))
        elif model_name == model_conf.LeNet1:  # LeNet1
            layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output, model.layers[6].output,
                      model.layers[8].output, ]
            # print(model.layers[1], model.layers[3], model.layers[6])
            layers = list(zip(4 * ['conv'] + 1 * ['dense'], layers))  #
        else:  # ResNet20
            layers = []
            for index in lst:
                layers.append(model.layers[index].output)
            layers = list(zip(19 * ['conv'] + 1 * ['dense'], layers))
    else:
        if model_name == model_conf.LeNet5:
            layers = [model.layers[1].output, model.layers[3].output, model.layers[4].output, model.layers[8].output,
                      model.layers[8].output, model.layers[9].output, model.layers[10].output]
            layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))
        elif model_name == model_conf.LeNet1:  # LeNet1
            layers = [model.layers[1].output, model.layers[3].output, model.layers[4].output, model.layers[6].output,
                      model.layers[8].output, ]
            layers = list(zip(4 * ['conv'] + 1 * ['dense'], layers))
        else:
            layers = []
            for index in lst:
                if index != len(model.layers) - 1:
                    layers.append(model.layers[index - 1].output)
                else:
                    layers.append(model.layers[index].output)
            layers = list(zip(19 * ['conv'] + 1 * ['dense'], layers))

    return input, layers, test, train, pred_test, true_test, pred_test_prob, Y_train


def gen_data_svhn(model_name, use_adv=True, deepxplore=False):
    model_path = model_conf.get_model_path(model_conf.svhn, model_name)  #
    (X_train, Y_train), (X_test, Y_test) = SVNH_DatasetUtil.load_data()
    Y_train = np.argmax(Y_train, axis=1)  # modify 原本是一堆矩阵,[0,0,0,1,0]代表第四类
    Y_test = np.argmax(Y_test, axis=1)
    if use_adv:
        attack_lst = ['cw', 'fgsm', 'jsma', 'bim']
        adv_image_all = []
        adv_label_all = []
        for attack in attack_lst:
            im, lab = model_conf.get_adv_path(attack, model_conf.svhn, model_name)
            adv_image_all.append(np.load(im))
            adv_label_all.append(np.load(lab))
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
    if not deepxplore:
        if model_name == model_conf.LeNet5:
            layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output, model.layers[6].output,
                      model.layers[8].output, model.layers[9].output, model.layers[10].output]
            layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))
        else:  # Vgg16
            layers = []
            for i in range(1, 19):
                layers.append(model.layers[i].output)
            for i in range(20, 23):
                layers.append(model.layers[i].output)
            layers = list(zip(18 * ['conv'] + 3 * ['dense'], layers))
    else:
        if model_name == model_conf.LeNet5:
            layers = [model.layers[1].output, model.layers[3].output, model.layers[4].output, model.layers[8].output,
                      model.layers[8].output, model.layers[9].output, model.layers[10].output]
            layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))
        else:
            layers = []
            for i in range(1, 19):
                layers.append(model.layers[i].output)
            for i in range(20, 23):
                layers.append(model.layers[i].output)
            layers = list(zip(18 * ['conv'] + 3 * ['dense'], layers))

    return input, layers, test, train, pred_test, true_test, pred_test_prob, Y_train


# 废弃
def exp_dsc():
    input, layers, test, train, pred_test, true_test, pred_test_prob, Y_train = gen_data_mnist(False)
    dsc = metrics.DSC(train, Y_train, input, layers)
    rate = dsc.fit(test, pred_test)
    score = dsc.get_sore()
    df = pd.DataFrame([])
    df["DSA"] = score
    df.to_csv('./all_output/output_mnist/DSC_{}.csv'.format(rate))


def exp_sa(model_name, coverage, gen_data, index=-1, use_adv=False, name="no", deepxplore=False):
    # 初始化变量
    k_bins = 1000
    if use_adv:
        dataset = name + '_adv'
    else:
        dataset = name
    # 计算覆盖
    input, layers, test, train, pred_test, true_test, pred_test_prob, Y_train = gen_data(model_name, use_adv=use_adv,
                                                                                         deepxplore=deepxplore)

    # print("test len")
    # print(len(test))
    model = None
    rate = None
    rank_lst_time = None
    rank_lst2_time = None
    st = time.time()  # 计算排序时间
    if coverage == "LSC":
        model = metrics.LSC(train, Y_train, input, [layers[index]], k_bins=k_bins, u=100)
        rate = model.fit(test, pred_test)
        # model.get_rate(auto=True)  # 获得覆盖率
    elif coverage == "DSC":
        print("dsc..")
        model = metrics.DSC(train, Y_train, input, layers, k_bins=k_bins)
        print("dsc..  fit...")
        rate = model.fit(test, pred_test)
        #  model.get_rate()  # 获得覆盖率
    en = time.time()
    pre_time = st - en
    start = time.time()
    rank_lst2 = model.rank_2()
    end = time.time()
    rank_lst2_time = start - end + pre_time
    start = time.time()
    rank_lst = model.rank_fast()
    end = time.time()
    rank_lst_time = start - end + pre_time
    # score = model.get_sore()  # 获得分数
    u = model.get_u()  # 获得上界
    # 构造结果
    df = pd.DataFrame([])
    # df["LSA"] = score
    # print(pred_test)

    df['right'] = (pred_test == true_test).astype('int')  # right
    df['cam'] = 0  # cam
    df['cam'].loc[rank_lst] = list(range(1, len(rank_lst) + 1))  # cam
    df['ctm'] = 0
    df['ctm'].loc[rank_lst2] = list(range(1, len(rank_lst2) + 1))  # ctm
    df['rate'] = rate  # tate
    df['cam_time'] = rank_lst_time
    df['ctm_time'] = rank_lst2_time
    if rate is None:
        df["overtime"] = 1
    # 数据集_覆盖方法_分箱_上界_选择的层数
    if coverage == "LSC":
        # df['rate_auto'] = model.get_rate(auto=True)  # 使用auto
        # df['rate_auto_u'] = model.get_u()
        df.to_csv('./all_output/{}/{}/{}_{}_k_{}_u_{}_L_{}.csv'.format("output_" + name, model_name, dataset, coverage,
                                                                       k_bins, int(u),
                                                                       index))
    elif coverage == "DSC":
        df.to_csv(
            './all_output/{}/{}/{}_{}_k_{}_u_{}.csv'.format("output_" + name, model_name, dataset, coverage, k_bins, u))
    return start - end


def exec():
    ###################################################################LSC

    print("LSC mnist")
    t = exp_sa(model_conf.LeNet1, "LSC", gen_data_mnist, name="mnist")
    t = exp_sa(model_conf.LeNet1, "LSC", gen_data_mnist, name="mnist", use_adv=True)
    t = exp_sa(model_conf.LeNet5, "LSC", gen_data_mnist, name="mnist")
    t = exp_sa(model_conf.LeNet5, "LSC", gen_data_mnist, name="mnist", use_adv=True)
    print("LSC cifar")
    t = exp_sa(model_conf.vgg16, "LSC", gen_data_cifar, name="cifar")
    t = exp_sa(model_conf.vgg16, "LSC", gen_data_cifar, name="cifar", use_adv=True)
    t = exp_sa(model_conf.resNet20, "LSC", gen_data_cifar, name="cifar")
    t = exp_sa(model_conf.resNet20, "LSC", gen_data_cifar, name="cifar", use_adv=True)
    print("LSC fashion")
    t = exp_sa(model_conf.resNet20, "LSC", gen_data_fashion, name="fashion")
    t = exp_sa(model_conf.resNet20, "LSC", gen_data_fashion, name="fashion", use_adv=True)
    t = exp_sa(model_conf.LeNet1, "LSC", gen_data_fashion, name="fashion")
    t = exp_sa(model_conf.LeNet1, "LSC", gen_data_fashion, name="fashion", use_adv=True)
    print("LSC svhn,")
    t = exp_sa(model_conf.vgg16, "LSC", gen_data_svhn, name="svhn")
    t = exp_sa(model_conf.vgg16, "LSC", gen_data_svhn, name="svhn", use_adv=True)
    t = exp_sa(model_conf.LeNet5, "LSC", gen_data_svhn, name="svhn")
    t = exp_sa(model_conf.LeNet5, "LSC", gen_data_svhn, name="svhn", use_adv=True)
    print("LSC over")

    ###################################################################DSC

    print("DSC mnist")
    t = exp_sa(model_conf.LeNet1, "DSC", gen_data_mnist, name="mnist")  # mnist_let1
    t = exp_sa(model_conf.LeNet1, "DSC", gen_data_mnist, name="mnist", use_adv=True)
    t = exp_sa(model_conf.LeNet5, "DSC", gen_data_mnist, name="mnist")  # mnist_let5
    t = exp_sa(model_conf.LeNet5, "DSC", gen_data_mnist, name="mnist", use_adv=True)

    print("DSC cifar")
    t = exp_sa(model_conf.vgg16, "DSC", gen_data_cifar, name="cifar")  # cifar_vgg16
    t = exp_sa(model_conf.vgg16, "DSC", gen_data_cifar, name="cifar", use_adv=True)
    t = exp_sa(model_conf.resNet20, "DSC", gen_data_cifar, name="cifar")  # cifar_resNet20
    t = exp_sa(model_conf.resNet20, "DSC", gen_data_cifar, name="cifar", use_adv=True)

    print("DSC fashion")
    t = exp_sa(model_conf.resNet20, "DSC", gen_data_fashion, name="fashion")  # fashion_resNet20
    t = exp_sa(model_conf.resNet20, "DSC", gen_data_fashion, name="fashion", use_adv=True)
    t = exp_sa(model_conf.LeNet1, "DSC", gen_data_fashion, name="fashion")  # fashion_LetNet1
    t = exp_sa(model_conf.LeNet1, "DSC", gen_data_fashion, name="fashion", use_adv=True)  # fashion_LetNet1

    print("DSC svhn")
    # t = exp_sa(model_conf.vgg16, "DSC", gen_data_svhn, name="svhn")  # svhn_vgg16 # 超时
    # t = exp_sa(model_conf.vgg16, "DSC", gen_data_svhn, name="svhn", use_adv=True)  # 超时
    t = exp_sa(model_conf.LeNet5, "DSC", gen_data_svhn, name="svhn")  # svhn_LeNet5
    t = exp_sa(model_conf.LeNet5, "DSC", gen_data_svhn, name="svhn", use_adv=True)

    print("DSC over")


if __name__ == '__main__':
    exec()
