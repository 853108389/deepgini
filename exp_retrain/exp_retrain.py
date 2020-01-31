#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# 原始4个模型
import multiprocessing
from keras import Model, Input
import sys
import time
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import SVNH_DatasetUtil
import os
import matplotlib.pyplot as plt
# 初始化基本模型
import metrics
import gc
# 配置类,统一配置,暂时不用
import model_conf
from keras import backend as K
from scipy import stats

plt.switch_backend('agg')


def get_layers(model, dataset_name, model_name, deepxplore=False):
    input = model.layers[0].output
    if dataset_name == model_conf.mnist:
        if not deepxplore:  #
            if model_name == model_conf.LeNet5:
                layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output,
                          model.layers[6].output,
                          model.layers[8].output, model.layers[9].output, model.layers[10].output]
                layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))  # 激活,池化,全连接
            else:
                layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output,
                          model.layers[6].output,
                          model.layers[8].output, ]
                # print(model.layers[1], model.layers[3], model.layers[6])
                layers = list(zip(4 * ['conv'] + 1 * ['dense'], layers))  #
        else:
            if model_name == model_conf.LeNet5:
                layers = [model.layers[1].output, model.layers[3].output, model.layers[4].output,
                          model.layers[6].output,
                          model.layers[8].output, model.layers[9].output, model.layers[10].output]
                layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))  # 卷积,池化,全连接
            else:
                layers = [model.layers[1].output, model.layers[3].output, model.layers[4].output,
                          model.layers[6].output,
                          model.layers[8].output, ]
                layers = list(zip(4 * ['conv'] + 1 * ['dense'], layers))
    elif dataset_name == model_conf.fashion:
        lst = []
        for index, layer in enumerate(model.layers):
            if 'activation' in layer.name:
                lst.append(index)
        lst.append(len(model.layers) - 1)
        if not deepxplore:
            if model_name == model_conf.LeNet5:  # 选择模型
                layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output,
                          model.layers[6].output,
                          model.layers[8].output, model.layers[9].output, model.layers[10].output]
                layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))
            elif model_name == model_conf.LeNet1:  # LeNet1
                layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output,
                          model.layers[6].output,
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
                layers = [model.layers[1].output, model.layers[3].output, model.layers[4].output,
                          model.layers[8].output,
                          model.layers[8].output, model.layers[9].output, model.layers[10].output]
                layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))
            elif model_name == model_conf.LeNet1:  # LeNet1
                layers = [model.layers[1].output, model.layers[3].output, model.layers[4].output,
                          model.layers[6].output,
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
    elif dataset_name == model_conf.svhn:
        if not deepxplore:
            if model_name == model_conf.LeNet5:
                layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output,
                          model.layers[6].output,
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
                layers = [model.layers[1].output, model.layers[3].output, model.layers[4].output,
                          model.layers[8].output,
                          model.layers[8].output, model.layers[9].output, model.layers[10].output]
                layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))
            else:
                layers = []
                for i in range(1, 19):
                    layers.append(model.layers[i].output)
                for i in range(20, 23):
                    layers.append(model.layers[i].output)
                layers = list(zip(18 * ['conv'] + 3 * ['dense'], layers))
    else:
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
    return input, layers


def exp(model, name, model_name, X_train, Y_train, X_test, Y_test, deep_metric, load_exist_table=True,
        **kwargs):
    pred_test_prob = model.predict(X_test)
    pred_test = np.argmax(pred_test_prob, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    rank_lst2 = None
    rank_lst = None
    rank_lst_time = None
    rank_lst2_time = None
    rate = 0
    if len(kwargs) != 0:
        params = "_" + "_".join([str(k) + "_" + str(v) for k, v in kwargs.items()])
    else:
        params = ""
    path = './final_exp/res/{}/{}_{}_{}.csv'.format(name, name, model_name, deep_metric + params)
    if load_exist_table:
        # 如果有文件就不要在做实验了
        if os.path.exists(path):
            print("load_exist_table")
            df = pd.read_csv(path, index_col=0)
            return df
    df = pd.DataFrame([])
    if deep_metric == "deepgini":
        rank_lst = metrics.deep_metric(pred_test_prob)
    elif deep_metric == "deepgini2" or deep_metric == "deepgini3":
        rank_lst, score = metrics.deep_metric2(pred_test_prob)
        df["score"] = score
    else:
        if deep_metric == "nac":
            input, layers, = get_layers(model, name, model_name, deepxplore=True)
        else:
            input, layers, = get_layers(model, name, model_name, deepxplore=False)
        if deep_metric == "nac":
            ac = metrics.nac(X_test, input, layers, t=kwargs["t"])
            rate = ac.fit()
            start = time.time()
            rank_lst = ac.rank_fast(X_test)
            end = time.time()
            rank_lst_time = start - end

            start = time.time()
            rank_lst2 = ac.rank_2(X_test)
            end = time.time()
            rank_lst2_time = start - end

        if deep_metric == 'kmnc':
            # 只能贪心排
            km = metrics.kmnc(X_train, input, layers, k_bins=kwargs["k"])
            start = time.time()
            rank_lst = km.rank_fast(X_test)
            end = time.time()
            rank_lst_time = start - end
            rate = km.fit(X_test)
        elif deep_metric == 'nbc':
            # 可以贪心排
            # 可以单个样本比较排
            # 0 0.5 1
            bc = metrics.nbc(X_train, input, layers, std=kwargs["std"])
            start = time.time()
            rank_lst = bc.rank_fast(X_test, use_lower=True)
            end = time.time()
            rank_lst_time = start - end
            start = time.time()
            rank_lst2 = bc.rank_2(X_test, use_lower=True)
            end = time.time()
            rank_lst2_time = start - end
            rate = bc.fit(X_test, use_lower=True)

        elif deep_metric == 'snac':
            # 可以贪心排
            # 可以单个样本比较排
            # 0 0.5 1
            bc = metrics.nbc(X_train, input, layers, std=kwargs["std"])
            start = time.time()
            rank_lst = bc.rank_fast(X_test, use_lower=False)
            end = time.time()
            rank_lst_time = start - end
            start = time.time()
            rank_lst2 = bc.rank_2(X_test, use_lower=False)
            end = time.time()
            rank_lst2_time = start - end
            rate = bc.fit(X_test, use_lower=False)
        elif deep_metric == 'tknc':
            # 只能贪心排
            # 1 2 3
            tk = metrics.tknc(X_test, input, layers, k=kwargs["k"])
            start = time.time()
            rank_lst = tk.rank(X_test)
            end = time.time()
            rank_lst_time = start - end
            rate = tk.fit(list(range(len(X_test))))
        elif deep_metric == 'lsc':
            label = np.argmax(Y_train, axis=1)
            st = time.time()  # 计算排序时间
            model = metrics.LSC(X_train, label, input, [layers[kwargs["index"]]], u=100)
            rate = model.fit(X_test, pred_test)
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
        elif deep_metric == "dsc":
            label = np.argmax(Y_train, axis=1)
            st = time.time()  # 计算排序时间
            model = metrics.DSC(X_train, label, input, layers)
            rate = model.fit(X_test, pred_test)
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

    df['right'] = (pred_test == Y_test).astype('int')
    df['cam'] = 0
    df['ctm'] = 0
    df['cam_time'] = rank_lst_time
    df['ctm_time'] = rank_lst2_time
    if rank_lst is not None:
        df['cam'].loc[rank_lst] = list(range(1, len(rank_lst) + 1))
    if rank_lst2 is not None:
        df['ctm'].loc[rank_lst2] = list(range(1, len(rank_lst2) + 1))
    df['rate'] = rate

    if deep_metric == "random":
        df = df.sample(frac=1, random_state=41).reset_index()
    df.to_csv(path)
    return df


def df_process(df, ascending=True):
    # 修改列名
    df["case_index"] = df.index
    # rank_arr = df["cam"]
    # 获得用例集大小
    n = df.shape[0]
    cols_arr = list(df)
    df_dict = {}
    if "cam" in cols_arr and (df["cam"] != 0).any() and len(df["cam"]) != 0:  # TODO: all 和 any
        # 按照cam大小排序
        df_case_rank = df.sort_values(axis=0, by=["cam"], ascending=ascending)
        # 重新设置索引
        df_case_rank = df_case_rank.reset_index(drop=True)
        df_dict["cam"] = df_case_rank
    if "ctm" in cols_arr and (df["ctm"] != 0).any() and len(df["ctm"]) != 0:
        # 按照ctm大小排序
        df_case_rank = df.sort_values(axis=0, by=["ctm"], ascending=ascending)
        # 重新设置索引
        df_case_rank = df_case_rank.reset_index(drop=True)
        df_dict["ctm"] = df_case_rank
    if "index" in cols_arr and (df["index"] != 0).any() and len(df["index"]) != 0:
        # 按照index大小排序
        df_case_rank = df.sort_values(axis=0, by=["index"], ascending=ascending)
        # 重新设置索引
        df_case_rank = df_case_rank.reset_index(drop=True)
        df_dict["random"] = df_case_rank
    return df_dict


def get_ts(n=None, k=10, u=100, ):
    # th_per_arr = [0, 1, 2, 3, 4, 5]
    # th_per_arr = [0, 1, 5, 10, 20, 50]
    # th_per_arr = [0, 0.1, 0.5, 1, 5, 10]
    th_per_arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # th_per_arr = [0, 1, 3, 5, 10, 20]
    # 创建阈值数组
    if n is None:
        # 返回百分比数值数组
        threshold_arr = th_per_arr
        # threshold_arr = [x for x in range(0, u) if x % k == 0 or x == 1]
    else:
        # 返回具体的个数
        threshold_arr = [n * x // 100 for x in th_per_arr]
        # threshold_arr = [n * x // 100 for x in range(0, u) if x % k == 0 or x == 1]
    return threshold_arr


def model_fit(model, path, X_train, Y_train, X_val, Y_val, name=None, ts=None, batch_size=128, nb_epoch=10, verbose=1,
              plot_his=True):
    checkpoint = ModelCheckpoint(filepath=path, monitor='val_accuracy', mode='auto',
                                 save_best_only='True')
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(X_val, Y_val),
                        callbacks=[checkpoint], verbose=verbose)  # modify
    model = load_model(path)
    score = model.evaluate(X_val, Y_val, verbose=0)  # modify
    return score


def my_key(s: str):
    ls = ["deepgini", "random", "lsc", "LSC", "dsc", "DSC", "nac_t_0", "nac_t_0.75", "nac",
          "kmnc",
          "nbc_std_0", "nbc_std_0.5", "nbc_std_1", "nbc",
          "snac_std_0", "snac_std_0.5", "snac_std_1", "snac",
          "tknc_k_1", "tknc_k_2", "tknc_k_3", "tknc",
          "ssc", "svc", "vsc", "vvc", ]
    res = [s.find(x) > -1 for x in ls]
    return res[::-1]


def plot_curve_with_metric(name, model_name):
    color_dict = {}
    dict = {
        "deepgini": "crimson",
        "deepgini2": "darkgreen",
        "deepgini3": "gold",
        "random": "black",
        "lsc": "SteelBlue",
        "dsc": "MediumSpringGreen",
        "nac": "yellowgreen",
        "nbc": "Chartreuse",
        "snac": 'Violet',
        "tknc": "SlateGray",
        "kmnc": "skyblue"
    }
    for k, v in dict.items():
        color_dict[k] = v
    base_path = './final_exp/res/{}/{}/'.format(name, model_name)
    # base_path = './final_exp/res/{}/'.format(name)
    list1 = os.listdir(base_path)
    list1.sort(key=my_key)
    # print(list1)
    ts_arr = get_ts()
    plt.xlabel("percentage of test cases")
    plt.ylabel("acc")
    # plt.xticks(range(6), range(6))
    plt.xticks(range(len(ts_arr)), ts_arr)

    for p in list1:
        args_arr = p.split("_")
        al = 0.5
        c = dict[args_arr[0].lower()]
        ls = "--"
        if p.endswith("random.csv"):
            ls = "-"
            al = 1
        if args_arr[0] == "deepgini":
            ls = "-"
            al = 1
        # print("{}{}".format(base_path, p))
        df_res = pd.read_csv("{}{}".format(base_path, p))
        plt.plot(range(1, len(df_res["ts"])), df_res["acc"][1:], label=p, alpha=al, color=c)
        ts = df_res["ts"]
    plt.legend()
    plt.title(model_conf.get_pair_name(name, model_name))
    plt.savefig('./final_exp/fig/{}_{}_metrics.png'.format(name, model_name))
    plt.close()


def gen_data_mnist(nb_classes=10):
    # 加载数据集
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 28*28
    X_train = X_train.astype('float32').reshape(-1, 28, 28, 1)
    X_test = X_test.astype('float32').reshape(-1, 28, 28, 1)
    X_train /= 255
    X_test /= 255
    # 数据预处理
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test,


def gen_data_fashion(nb_classes=10, ):
    # 加载数据集
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()  # 28*28
    X_train = X_train.astype('float32').reshape(-1, 28, 28, 1)
    X_test = X_test.astype('float32').reshape(-1, 28, 28, 1)
    X_train /= 255
    X_test /= 255
    # 数据预处理
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test,


def gen_data_cifar(nb_classes=10, ):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 28*28
    X_train = X_train.astype('float32').reshape(-1, 32, 32, 3)
    X_test = X_test.astype('float32').reshape(-1, 32, 32, 3)
    X_train /= 255
    X_test /= 255
    Y_train = Y_train.reshape(-1)
    Y_test = Y_test.reshape(-1)

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test,


def gen_data_svhn(nb_classes=10):
    (X_train, Y_train), (X_test, Y_test) = SVNH_DatasetUtil.load_data()  # 32*32
    Y_test = np.argmax(Y_test, axis=1)
    Y_train = np.argmax(Y_train, axis=1)
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test,


def gen_data(name, model_name, nb_classes=10):
    X_train, X_val, X_test, Y_train, Y_val, Y_test = None, None, None, None, None, None

    if name.startswith("mnist"):
        X_train, X_test, Y_train, Y_test, = gen_data_mnist()  # 获得原始数据集
    elif name.startswith("fashion"):
        X_train, X_test, Y_train, Y_test, = gen_data_fashion()  # 获得原始数据集
    elif name.startswith("svhn"):
        X_train, X_test, Y_train, Y_test, = gen_data_svhn()  # 获得原始数据集
    elif name.startswith("cifar"):
        X_train, X_test, Y_train, Y_test, = gen_data_cifar()  # 获得原始数据集
    attack_lst = ['cw', 'fgsm', 'jsma', 'bim']

    adv_image_test_arr = []
    adv_label_test_arr = []
    for attack in attack_lst:
        im, lab = model_conf.get_adv_path(attack, name, model_name)
        attack_test = np.load(im)
        attack_lable = np.load(lab)
        adv_image_test_arr.append(attack_test)
        adv_label_test_arr.append(attack_lable)

    adv_X_test = np.concatenate(adv_image_test_arr, axis=0)
    adv_Y_test = np.concatenate(adv_label_test_arr, axis=0)

    adv_Y_test = np_utils.to_categorical(adv_Y_test, nb_classes)
    X_test = adv_X_test
    Y_test = adv_Y_test

    # 1w条数据
    # adv_X_test, _, adv_Y_test, _ = train_test_split(adv_X_test, adv_Y_test, train_size=5000, random_state=42)
    # X_test, _, Y_test, _ = train_test_split(X_test, Y_test, train_size=5000, random_state=42)
    # X_test = np.r_[X_test, adv_X_test]  # 共1w条
    # Y_test = np.r_[Y_test, adv_Y_test]

    # adv_X_test, _, adv_Y_test, _ = train_test_split(adv_X_test, adv_Y_test, train_size=5000)
    # X_test, _, Y_test, _ = train_test_split(X_test, Y_test, train_size=5000)
    # adv_Y_test = np_utils.to_categorical(adv_Y_test, nb_classes)
    # X_test = np.concatenate([X_test, adv_X_test], axis=0)  # 共1w条 2000:8000
    # Y_test = np.concatenate([Y_test, adv_Y_test], axis=0)
    return X_train, X_test, Y_train, Y_test


def exec(dataset_name, model_name, deep_metric, **kwargs):
    path = "./final_exp/model/{}/{}/model_mnist_ts_{}_{}.hdf5"  # 模型储存路径
    model_path = model_conf.get_model_path(dataset_name, model_name)
    # 加载数据集
    X_train, X_test, Y_train, Y_test = gen_data(dataset_name, model_name)
    print('Train:{},Test:{}'.format(len(X_train), len(X_test)))
    print(deep_metric, kwargs)

    if len(kwargs) != 0:
        params = "_" + "_".join([str(k) + "_" + str(v) for k, v in kwargs.items()])
    else:
        params = ""
    ascending = True
    # 获取阈值
    threshold_arr = get_ts(len(X_test))  # 根据测试用例个数获取阈值
    ts_percent_arr = get_ts()
    # #进行试验

    # print("params are ", kwargs)
    #
    for ixx in range(2):  # 对于cam 和 ctm
        pre_model_path = model_path  # 初始化模型路径
        cur_acc_arr = []  # 初始化结果数组
        for i, ts in enumerate(threshold_arr):  # 对于所有的阈值
            ts = int(ts)
            pre_model = load_model(pre_model_path)  # 每次都重新加载上一次的模型
            if ts == 0:
                pre_acc = pre_model.evaluate(X_test, Y_test, verbose=0)[1]  # 初始化精度
                cur_acc_arr.append(pre_acc)  # 添加初始精度
                print("origin acc is {}".format(pre_acc))
                continue
            # 做实验
            print("exp: {} .....".format(i))
            df = exp(pre_model, dataset_name, model_name, X_train, Y_train, X_test, Y_test, deep_metric,
                     load_exist_table=load_exist_table, **kwargs)
            print("exp over...")

            # 数据处理
            df_case_rank_dict = df_process(df, ascending=ascending)
            print("{} has {} rank method: {}".format(deep_metric, len(df_case_rank_dict.keys()),
                                                     df_case_rank_dict.keys()))
            if ixx == 0 and "cam" in df_case_rank_dict.keys():
                key = "cam"
                df_case_rank = df_case_rank_dict["cam"]
            elif ixx == 1 and "ctm" in df_case_rank_dict.keys():
                key = "ctm"
                df_case_rank = df_case_rank_dict["ctm"]
            elif ixx == 0 and "random" in df_case_rank_dict.keys():
                key = "random"
                df_case_rank = df_case_rank_dict["random"]
            else:
                break
            # if deep_metric == "dsc" and key == "cam":
            #     continue
            # if deep_metric == "lsc" and key == "cam":
            #     continue
            if deep_metric == "deepgini2":  # 采样获得用例的index
                kernel = stats.gaussian_kde(df["score"])
                val = kernel.evaluate(df["score"])  # 获得函数
                pdf = val / sum(val)  # 进行归一化
                np.random.seed(42)
                case_index = np.random.choice(df.index, p=pdf, size=ts)  # 采样获得用例
            elif deep_metric == "deepgini3":  # 采样获得用例的index
                kernel = stats.gaussian_kde(df["score"])
                val = kernel.evaluate(df["score"])  # 获得函数
                val2 = val * df["score"].values  # 乘以权重
                pdf = val2 / sum(val2)  # 进行归一化
                np.random.seed(42)
                case_index = np.random.choice(df.index, p=pdf, size=ts)  # 采样获得用例
                print("===================init cases over=====================")
            else:
                case_index = df_case_rank[:ts]["case_index"]  # 按阈值筛选用例
            add_test_X = X_test[case_index]
            add_test_Y = Y_test[case_index]
            if only_add:
                X_train_now = add_test_X
                Y_train_now = add_test_Y
            else:
                X_train_now = np.r_[X_train, add_test_X]
                Y_train_now = np.r_[Y_train, add_test_Y]

            print('{}, 添加了{}个用例,现在的训练集长度为: {}'.format(i, ts, len(X_train_now)))
            # path = "./final_exp/model/{}/{}/model_mnist_ts_{}_{}.hdf5"  # 模型储存路径
            path = path.format(dataset_name, model_name, ts_percent_arr[i], deep_metric + params + "_" + key)
            if load_exist_model and os.path.exists(path):
                model = load_model(path)
                cur_acc = model.evaluate(X_test, Y_test, verbose=0)[1]
            else:
                print("pre path is {} ,now path is {}".format(pre_model_path, path))
                cur_acc = model_fit(pre_model, path, X_train_now, Y_train_now, X_test, Y_test, batch_size=128,
                                    name=dataset_name, ts=ts_percent_arr[i], verbose=1, nb_epoch=10)[1]
            print("pre_acc is {} , cur_acc is {} ,improve {}".format(pre_acc, cur_acc, (cur_acc - pre_acc)))
            cur_acc_arr.append(cur_acc)
            pre_acc = cur_acc
            pre_model_path = path
            K.clear_session()
        if len(cur_acc_arr) == len(ts_percent_arr):  # 如果执行了cam/ctm,才记录结果
            print(cur_acc_arr)
            df_res = pd.DataFrame()
            df_res["ts"] = ts_percent_arr
            df_res["acc"] = cur_acc_arr
            res_path = "./final_exp/res/{}/{}/{}_{}.csv".format(dataset_name, model_name, deep_metric + params, key)
            df_res.to_csv(res_path)


# def exec2(dataset_name, model_name, deep_metric, **kwargs):
#     ascending = True
#     path = "./final_exp/model/{}/{}/model_mnist_ts_{}_{}.hdf5"  # 模型储存路径
#     model_path = model_conf.get_model_path(dataset_name, model_name)
#     # 加载数据集
#     X_train, X_test, Y_train, Y_test = gen_data(dataset_name, model_name)
#     print('Train:{},Test:{}'.format(len(X_train), len(X_test)))
#     print(deep_metric, kwargs)
#     ##### 模型
#     # 加载原始模型
#     origin_model = load_model(model_path)
#     df = exp(origin_model, dataset_name, model_name, X_train, Y_train, X_test, Y_test, deep_metric,
#              load_exist_table=load_exist_table, **kwargs)
#     df_case_rank_dict = df_process(df, ascending=ascending)
#     print("{} has {} rank method: {}".format(deep_metric, len(df_case_rank_dict.keys()),
#                                              df_case_rank_dict.keys()))
#     # print("params are ", kwargs)
#     #
#
#     pdf = stats.gaussian_kde(df["score"])
#
#     import seaborn as sns
#     sns.distplot(df["score"])
#     plt.savefig('./final_exp/fig/{}_{}_dis.png'.format(dataset_name, model_name))
#     plt.close()
#     # for key, df_case_rank in df_case_rank_dict.items():
#     #     case_index = df_case_rank["case_index"]
#     #     case_index

def main(dataset_name, model_name):
    if (dataset_name == model_conf.mnist) or (dataset_name == model_conf.fashion and model_name == model_conf.LeNet1):
        print("kmnc...")
        deep_metric = "kmnc"
        kwargs = {"k": 1000}
        exec(dataset_name, model_name, deep_metric, **kwargs)

    print("==aaa===")
    # deep_metric = "deepgini3"  # 采样,乘以权重
    # kwargs = {}
    # exec(dataset_name, model_name, deep_metric, **kwargs)
    # deep_metric = "deepgini2"  # 采样,没称权重,
    # kwargs = {}
    # exec(dataset_name, model_name, deep_metric, **kwargs)
    # deep_metric = "deepgini"
    # kwargs = {}
    # exec(dataset_name, model_name, deep_metric, **kwargs)
    #
    # # # #
    # deep_metric = "nac"
    # kwargs = {"t": 0}
    # exec(dataset_name, model_name, deep_metric, **kwargs)
    # deep_metric = "nac"
    # kwargs = {"t": 0.75}
    # exec(dataset_name, model_name, deep_metric, **kwargs)
    # # #
    # deep_metric = "tknc"
    # kwargs = {"k": 1}
    # exec(dataset_name, model_name, deep_metric, **kwargs)
    # deep_metric = "tknc"
    # kwargs = {"k": 2}
    # exec(dataset_name, model_name, deep_metric, **kwargs)
    # deep_metric = "tknc"
    # kwargs = {"k": 3}
    # exec(dataset_name, model_name, deep_metric, **kwargs)
    # #
    # deep_metric = "nbc"
    # kwargs = {"std": 0}
    # exec(dataset_name, model_name, deep_metric, **kwargs)
    # deep_metric = "nbc"
    # kwargs = {"std": 0.5}
    # exec(dataset_name, model_name, deep_metric, **kwargs)
    # deep_metric = "nbc"
    # kwargs = {"std": 1}
    # exec(dataset_name, model_name, deep_metric, **kwargs)
    # # #
    # deep_metric = "snac"
    # kwargs = {"std": 0}
    # exec(dataset_name, model_name, deep_metric, **kwargs)
    # deep_metric = "snac"
    # kwargs = {"std": 0.5}
    # exec(dataset_name, model_name, deep_metric, **kwargs)
    if dataset_name == model_conf.cifar10:
        deep_metric = "snac"
        kwargs = {"std": 1}
        exec(dataset_name, model_name, deep_metric, **kwargs)
    # #
    # deep_metric = "lsc"
    # kwargs = {"index": -1}
    # exec(dataset_name, model_name, deep_metric, **kwargs)
    # deep_metric = "dsc"
    # kwargs = {}
    # exec(dataset_name, model_name, deep_metric, **kwargs)
    #
    # deep_metric = "random"
    # kwargs = {}
    # exec(dataset_name, model_name, deep_metric, **kwargs)

    plot_curve_with_metric(dataset_name, model_name)


# 表格打印
def statistic_res(name, model_name):
    base_path = './final_exp/res/{}/{}'.format(name, model_name)
    list1 = os.listdir(base_path)
    df_acc = pd.DataFrame()

    for i, p in enumerate(list1):
        df_res = pd.read_csv("{}{}".format(base_path, p))
        p = os.path.splitext(p)[0]
        args_arr = p.split("_")
        method_name = args_arr[-1]
        metric_name = "_".join(args_arr[:-2])
        col_name = metric_name + "_" + str(method_name)
        # print(col_name)
        if i == 0:
            ts = df_res["ts"]
            df_acc["ts"] = ts
        df_acc[col_name] = df_res["acc"]
    df_acc.to_csv("./final_exp/res/{}/{}_{}_acc.csv".format(name, name, model_name))
    # df_adv.to_csv("{}{}_adv.csv".format("./model_retrain/res/", model_conf.get_pair_name(name, model_name)))


# 初始化文件夹和模型
def init():
    model_data_dict = {
        model_conf.mnist: [model_conf.LeNet5, model_conf.LeNet1],
        model_conf.fashion: [model_conf.LeNet1, model_conf.resNet20],
        model_conf.cifar10: [model_conf.vgg16, model_conf.resNet20],
        model_conf.svhn: [model_conf.LeNet5, model_conf.vgg16]
    }
    print("init dir..")
    name_arr = ["fashion", "mnist", "svhn", "cifar"]
    data_arr = ["res", "fig", "model"]
    base_path = "./final_exp"
    mk_path_arr = []
    for x1 in data_arr:
        p = base_path + "/" + x1
        mk_path_arr.append(p)
        for name in name_arr:
            p2 = p + "/" + name
            mk_path_arr.append(p2)
            model_list = model_data_dict[name]
            for model_name in model_list:
                mk_path_arr.append(p2 + "/" + model_name)
    for x in mk_path_arr:
        if not os.path.exists(x):
            os.makedirs(x)
    print("init  dir over")
    print("init model")


if __name__ == '__main__':
    load_exist_table = False
    load_exist_model = False
    only_add = False
    #
    init()
    # ######数据集
    # mnist
    dataset_name = model_conf.mnist
    model_name = model_conf.LeNet1
    main(dataset_name, model_name)
    #
    dataset_name = model_conf.mnist
    model_name = model_conf.LeNet5
    main(dataset_name, model_name)
    #
    #  fashion
    dataset_name = model_conf.fashion
    model_name = model_conf.LeNet1
    main(dataset_name, model_name)
    dataset_name = model_conf.fashion
    model_name = model_conf.resNet20
    main(dataset_name, model_name)
    ####vgg16跑这里
    # # svhn
    dataset_name = model_conf.svhn
    model_name = model_conf.LeNet5
    main(dataset_name, model_name)
    dataset_name = model_conf.svhn
    model_name = model_conf.vgg16
    main(dataset_name, model_name)
    ####

    # # #
    # # # # cifar
    dataset_name = model_conf.cifar10
    model_name = model_conf.resNet20
    main(dataset_name, model_name)
    dataset_name = model_conf.cifar10
    model_name = model_conf.vgg16
    main(dataset_name, model_name)
