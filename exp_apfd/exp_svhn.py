#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os

import pandas as pd
import numpy as np
from keras.datasets import mnist, cifar10, cifar100
from keras.models import load_model
import metrics
import SVNH_DatasetUtil
import time
import model_conf


def gen_data(model_name, use_adv=True, deepxplore=False):
    model_path = model_conf.get_model_path(model_conf.svhn, model_name)  #
    (X_train, Y_train), (X_test, Y_test) = SVNH_DatasetUtil.load_data()
    Y_test = np.argmax(Y_test, axis=1)
    #  model_path = './model/model_svhn.hdf5'
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
        print("adv: ", len(adv_image_all))
        test = np.concatenate([X_test, adv_image_all], axis=0)
        true_test = np.concatenate([Y_test, adv_label_all], axis=0)
    else:
        test = X_test
        true_test = Y_test
    train = X_train

    model = load_model(model_path)
    model.summary()
    pred_test_prob = model.predict(test)
    pred_test = np.argmax(pred_test_prob, axis=1)
    input = model.layers[0].output
    # print(dir(model.layers[0]))
    # print(input)
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
    # print(len(layers), layers)
    return input, layers, test, train, pred_test, true_test, pred_test_prob


def exp(model_name, coverage, use_adv, std=0, k=1, k_bins=1000):
    input, layers, test, train, pred_test, true_test, pred_test_prob = gen_data(model_name, use_adv=use_adv)
    rank_lst2 = None
    rank_lst_time = None
    rank_lst2_time = None
    if coverage == 'kmnc':
        # 只能贪心排
        km = metrics.kmnc(train, input, layers, k_bins=k_bins)
        start = time.time()
        rank_lst = km.rank_fast(test)
        end = time.time()
        rank_lst_time = start - end
        if rank_lst is not None:
            rate = km.fit(test)
        # rate = km.fit(test)
    elif coverage == 'nbc':
        # 可以贪心排
        # 可以单个样本比较排
        # 0 0.5 1
        bc = metrics.nbc(train, input, layers, std=std)
        start = time.time()
        rank_lst = bc.rank_fast(test, use_lower=True)
        end = time.time()
        rank_lst_time = start - end
        start = time.time()
        rank_lst2 = bc.rank_2(test, use_lower=True)
        end = time.time()
        rank_lst2_time = start - end
        rate = bc.fit(test, use_lower=True)
    elif coverage == 'snac':
        # 可以贪心排
        # 可以单个样本比较排
        # 0 0.5 1
        bc = metrics.nbc(train, input, layers, std=std)
        start = time.time()
        rank_lst = bc.rank_fast(test, use_lower=False)
        end = time.time()
        rank_lst_time = start - end
        start = time.time()
        rank_lst2 = bc.rank_2(test, use_lower=False)
        end = time.time()
        rank_lst2_time = start - end
        rate = bc.fit(test, use_lower=False)
    elif coverage == 'tknc':
        # 只能贪心排
        # 1 2 3
        start = time.time()
        tk = metrics.tknc(test, input, layers, k=k)
        rank_lst = tk.rank(test)
        end = time.time()
        rank_lst_time = start - end
        rate = tk.fit(list(range(len(test))))

    df = pd.DataFrame([])
    df['right'] = (pred_test == true_test).astype('int')
    df['cam'] = 0
    df['ctm'] = 0
    df['cam'].loc[rank_lst] = list(range(1, len(rank_lst) + 1))
    df['cam_time'] = rank_lst_time
    df['ctm_time'] = rank_lst2_time
    if rank_lst2 is not None:
        df['ctm'].loc[rank_lst2] = list(range(1, len(rank_lst2) + 1))
    if use_adv:
        dataset = 'svhn_adv'
    else:
        dataset = 'svhn'
    df['rate'] = rate
    if coverage == 'kmnc':
        df.to_csv('./all_output/output_svhn/{}/{}_{}_k_bins_{}.csv'.format(model_name, dataset, coverage, k_bins))
    elif coverage == 'nbc':
        df.to_csv('./all_output/output_svhn/{}/{}_{}_std_{}.csv'.format(model_name, dataset, coverage, std))
    elif coverage == 'snac':
        df.to_csv('./all_output/output_svhn/{}/{}_{}_std_{}.csv'.format(model_name, dataset, coverage, std))
    elif coverage == 'tknc':
        df.to_csv('./all_output/output_svhn/{}/{}_{}_k_{}.csv'.format(model_name, dataset, coverage, k))


def exp_nac(model_name, use_adv, t):
    input, layers, test, train, pred_test, true_test, pred_test_prob = gen_data(model_name, use_adv=use_adv,
                                                                                deepxplore=True)

    # 可以贪心排
    # 可以单个样本比较排
    # 0 0.5 1
    ac = metrics.nac(test, input, layers, t=t)
    rate = ac.fit()
    start = time.time()
    rank_lst = ac.rank_fast(test)
    end = time.time()
    rank_lst_time = start - end

    start = time.time()
    rank_lst2 = ac.rank_2(test)
    end = time.time()
    rank_lst2_time = start - end

    df = pd.DataFrame([])

    df['right'] = (pred_test == true_test).astype('int')
    df['cam'] = 0
    df['ctm'] = 0
    df['cam'].loc[rank_lst] = list(range(1, len(rank_lst) + 1))
    df['cam_time'] = rank_lst_time
    df['ctm_time'] = rank_lst2_time
    if rank_lst2 is not None:
        df['ctm'].loc[rank_lst2] = list(range(1, len(rank_lst2) + 1))
    if use_adv:
        dataset = 'svhn_adv'
    else:
        dataset = 'svhn'
    df['rate'] = rate

    df.to_csv('./all_output/output_svhn/{}/{}_nac_t_{}.csv'.format(model_name, dataset, t))


def exp_deep_metric(model_name, use_adv):
    rank_lst2 = None
    rank_lst2_time = None
    input, layers, test, train, pred_test, true_test, pred_test_prob = gen_data(model_name, use_adv=use_adv,
                                                                                deepxplore=False)
    model_path = model_conf.get_model_path(model_conf.svhn, model_name)  #
    model = load_model(model_path)
    start = time.time()
    pred_test_prob = model.predict(test)
    rank_lst = metrics.deep_metric(pred_test_prob)
    end = time.time()
    rank_lst_time = start - end
    df = pd.DataFrame([])
    df['right'] = (pred_test == true_test).astype('int')
    df['cam'] = 0
    df['cam'].loc[rank_lst] = list(range(1, len(rank_lst) + 1))
    df['ctm'] = 0
    df['cam_time'] = rank_lst_time
    df['ctm_time'] = rank_lst2_time
    if rank_lst2 is not None:
        df['ctm'].loc[rank_lst2] = list(range(1, len(rank_lst2) + 1))
    df['rate'] = 0
    if use_adv:
        dataset = 'svhn_adv'
    else:
        dataset = 'svhn'
    df.to_csv('./all_output/output_svhn/{}/{}_deep_metric.csv'.format(model_name, dataset))


def exec(model_name):
    # pass
    dic = {}
    #
    # start = time.time()
    # exp_nac(model_name, use_adv=False, t=0)
    # end = time.time()
    # dic['mnist_nac_t_0'] = (start - end)
    #
    # start = time.time()
    # exp_nac(model_name, use_adv=True, t=0)
    # end = time.time()
    # dic['mnist_adv_nac_t_0'] = (start - end)
    #
    # start = time.time()
    # exp_nac(model_name, use_adv=False, t=0.75)
    # end = time.time()
    # dic['mnist_nac_t_0.75'] = (start - end)
    #
    # start = time.time()
    # exp_nac(model_name, use_adv=True, t=0.75)
    # end = time.time()
    # dic['mnist_adv_nac_t_0.75'] = (start - end)
    #
    exp(model_name, coverage='kmnc', use_adv=False, k_bins=1000)
    # # exp(coverage='kmnc',use_adv=False,k_bins=10000)
    # exp(model_name, coverage='kmnc', use_adv=True, k_bins=1000)
    # # exp(coverage='kmnc',use_adv=True,k_bins=10000)
    #
    # start = time.time()
    # exp_deep_metric(model_name, use_adv=False)
    # end = time.time()
    # dic['mnist_ours'] = (start - end)
    #
    # start = time.time()
    # exp_deep_metric(model_name, use_adv=True)
    # end = time.time()
    # dic['mnist_adv_ours'] = (start - end)

    # start = time.time()
    # exp(model_name, coverage='tknc', use_adv=False, k=1)
    # end = time.time()
    # dic['mnist_tknc_k_1'] = (start - end)
    #
    # start = time.time()
    # exp(model_name, coverage='tknc', use_adv=False, k=2)
    # end = time.time()
    # dic['mnist_tknc_k_2'] = (start - end)
    #
    # start = time.time()
    # exp(model_name, coverage='tknc', use_adv=False, k=3)
    # end = time.time()
    # dic['mnist_tknc_k_3'] = (start - end)
    #
    # start = time.time()
    # exp(model_name, coverage='tknc', use_adv=True, k=1)
    # end = time.time()
    # dic['mnist_adv_tknc_k_1'] = (start - end)
    #
    # start = time.time()
    # exp(model_name, coverage='tknc', use_adv=True, k=2)
    # end = time.time()
    # dic['mnist_adv_tknc_k_2'] = (start - end)
    #
    # start = time.time()
    # exp(model_name, coverage='tknc', use_adv=True, k=3)
    # end = time.time()
    # dic['mnist_adv_tknc_k_3'] = (start - end)

    # start = time.time()
    # exp(model_name, coverage='nbc', use_adv=False, std=0.5)
    # end = time.time()
    # dic['mnist_nbc_std_0.5'] = (start - end)
    #
    # start = time.time()
    # exp(model_name, coverage='nbc', use_adv=False, std=1)
    # end = time.time()
    # dic['mnist_nbc_std_1'] = (start - end)
    #
    # start = time.time()
    # exp(model_name, coverage='nbc', use_adv=False, std=0)
    # end = time.time()
    # dic['mnist_nbc_std_0'] = (start - end)
    #
    # start = time.time()
    # exp(model_name, coverage='nbc', use_adv=True, std=0.5)
    # end = time.time()
    # dic['mnist_adv_nbc_std_0.5'] = (start - end)
    #
    # start = time.time()
    # exp(model_name, coverage='nbc', use_adv=True, std=1)
    # end = time.time()
    # dic['mnist_adv_nbc_std_1'] = (start - end)
    #
    # start = time.time()
    # exp(model_name, coverage='nbc', use_adv=True, std=0)
    # end = time.time()
    # dic['mnist_adv_nbc_std_0'] = (start - end)

    # start = time.time()
    # exp(model_name, coverage='snac', use_adv=False, std=0.5)
    # end = time.time()
    # dic['mnist_snac_std_0.5'] = (start - end)
    #
    # start = time.time()
    # exp(model_name, coverage='snac', use_adv=False, std=1)
    # end = time.time()
    # dic['mnist_snac_std_1'] = (start - end)
    #
    # start = time.time()
    # exp(model_name, coverage='snac', use_adv=False, std=0)
    # end = time.time()
    # dic['mnist_snac_std_0'] = (start - end)
    #
    # start = time.time()
    # exp(model_name, coverage='snac', use_adv=True, std=0.5)
    # end = time.time()
    # dic['mnist_adv_snac_std_0.5'] = (start - end)
    #
    # start = time.time()
    # exp(model_name, coverage='snac', use_adv=True, std=1)
    # end = time.time()
    # dic['mnist_adv_snac_std_1'] = (start - end)
    #
    # start = time.time()
    # exp(model_name, coverage='snac', use_adv=True, std=0)
    # end = time.time()
    # dic['mnist_adv_snac_std_0'] = (start - end)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    x = "./all_output/output_svhn/vgg16/"
    if not os.path.exists(x):
        os.makedirs(x)
    # model_name = model_conf.vgg16
    # # model_name = model_conf.LeNet5
    # exec(model_name)
    # print("dataset", "model_name")
    # print(model_conf.svhn, model_name)
    # print("adv path")
    # print(model_conf.get_adv_path("cw", model_conf.svhn, model_name))

    # print("================================================================")
    x = "./all_output/output_svhn/LeNet5/"
    if not os.path.exists(x):
        os.makedirs(x)
    model_name = model_conf.LeNet5
    exec(model_name)
    print("dataset", "model_name")
    print(model_conf.svhn, model_name)
    print("adv path")
    print(model_conf.get_adv_path("cw", model_conf.svhn, model_name))
