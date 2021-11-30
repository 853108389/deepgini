#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import keras
from keras.models import load_model
import numpy as np
import foolbox
from tqdm import tqdm
from keras.datasets import mnist, fashion_mnist
from keras.datasets import cifar10
from keras.datasets import cifar100
import SVNH_DatasetUtil
import itertools
import model_conf
import warnings

warnings.filterwarnings("ignore")
import multiprocessing


def adv_func(x, y, model_path, dataset='mnist', attack='fgsm'):
    keras.backend.set_learning_phase(0)
    model = load_model(model_path)
    foolmodel = foolbox.models.KerasModel(model, bounds=(0, 1), preprocessing=(0, 1))
    if attack == 'cw':
        # attack = foolbox.attacks.L2BasicIterativeAttack(foolmodel)
        attack = foolbox.attacks.CarliniWagnerL2Attack(foolmodel)
    elif attack == 'fgsm':
        # FGSM
        attack = foolbox.attacks.GradientSignAttack(foolmodel)
    elif attack == 'bim':
        # BIM
        attack = foolbox.attacks.L1BasicIterativeAttack(foolmodel)
    elif attack == 'jsma':
        # JSMA
        attack = foolbox.attacks.SaliencyMapAttack(foolmodel)
        # CW
        # attack=foolbox.attacks.DeepFoolL2Attack(foolmodel)
    result = []
    if dataset == 'mnist':
        w, h = 28, 28
    elif dataset == 'cifar10':
        w, h = 32, 32
    else:
        return False
    for image in tqdm(x):
        try:
            # adv=attack(image.reshape(28,28,-1),label=y,steps=1000,subsample=10)
            # adv=attack(image.reshape(w,h,-1),y,epsilons=[0.01,0.1],steps=10)
            if attack != 'fgsm':
                adv = attack(image.reshape(w, h, -1), y)
                adv = attack(image.reshape(w, h, -1), y)
                adv = attack(image.reshape(w, h, -1), y)
            else:
                adv = attack(image.reshape(w, h, -1), y, [0.01, 0.1])

            if isinstance(adv, np.ndarray):
                result.append(adv)
                if dataset == 'cifar10' and len(result) == 200:  # 每个标签每种攻击生成200个,共10 * 200 *4 =8000个反例
                    break
                if dataset == 'svhn' and len(result) == 200:
                    break
            else:
                print('adv fail')
        except:
            pass
    return np.array(result)


def generate_mnist_sample(model_path, label, attack):
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 28*28
    X_train = X_train.astype('float32').reshape(-1, 28, 28, 1)
    X_test = X_test.astype('float32').reshape(-1, 28, 28, 1)
    X_train /= 255
    X_test /= 255
    image_org = X_test[Y_test == label]
    adv = adv_func(image_org, label, model_path=model_path, dataset='mnist', attack=attack)
    return adv


def generate_cifar_sample(model_path, label, attack):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32*32
    X_train = X_train.astype('float32').reshape(-1, 32, 32, 3)
    X_test = X_test.astype('float32').reshape(-1, 32, 32, 3)
    X_train /= 255
    X_test /= 255

    Y_train = Y_train.reshape(-1)
    Y_test = Y_test.reshape(-1)

    image_org = X_test[Y_test == label]

    adv = adv_func(image_org, label, model_path=model_path, dataset='cifar10', attack=attack)
    return adv


def generate_cifar100_sample(model_path, label, attack):
    (X_train, Y_train), (X_test, Y_test) = cifar100.load_data(label_mode='coarse')  # 32*32
    X_train = X_train.astype('float32').reshape(-1, 32, 32, 3)
    X_test = X_test.astype('float32').reshape(-1, 32, 32, 3)
    X_train /= 255
    X_test /= 255

    Y_train = Y_train.reshape(-1)
    Y_test = Y_test.reshape(-1)
    image_org = X_test[Y_test == label]

    adv = adv_func(image_org, label, model_path=model_path, dataset='cifar10', attack=attack)
    return adv


def generate_fashion_sample(model_path, label, attack):
    ##modify
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()  ### modify
    print(X_train.shape)
    X_train = X_train.astype('float32').reshape(-1, 28, 28, 1)
    X_test = X_test.astype('float32').reshape(-1, 28, 28, 1)
    X_train /= 255
    X_test /= 255
    ##modify
    # path = './fashion-mnist/data/fashion'
    # X_train, Y_train = mnist_reader.load_mnist(path, kind='train')
    # X_test, Y_test = mnist_reader.load_mnist(path, kind='t10k')
    # X_train = X_train.astype('float32').reshape(-1, 28, 28, 1)
    # X_test = X_test.astype('float32').reshape(-1, 28, 28, 1)
    # X_train /= 255
    # X_test /= 255

    image_org = X_test[Y_test == label]
    adv = adv_func(image_org, label, model_path=model_path, dataset='mnist', attack=attack)
    return adv


def generate_svhn_sample(model_path, label, attack):
    (X_train, Y_train), (X_test, Y_test) = SVNH_DatasetUtil.load_data()  # 32*32

    image_org = X_test[np.argmax(Y_test, axis=1) == label]

    adv = adv_func(image_org, label, model_path=model_path, dataset='cifar10', attack=attack)
    return adv


def generate_adv_sample(s, model_path, dataset, attack):
    if dataset == 'mnist':
        sample_func = generate_mnist_sample
    elif dataset == 'svhn':
        sample_func = generate_svhn_sample
    elif dataset == 'fashion':
        sample_func = generate_fashion_sample
    elif dataset == 'cifar10':
        sample_func = generate_cifar_sample
    elif dataset == 'cifar20':
        sample_func = generate_cifar100_sample
    else:
        print('erro')
        return
    image = []
    label = []
    for i in range(10):
        print(i, "=======================")
        adv = sample_func(model_path, label=i, attack=attack)
        temp_image = adv
        temp_label = i * np.ones(len(adv))
        # np.save('./temp/{}_{}_image{}'.format(s, attack, i), image)
        # np.save('./temp/{}_{}_label{}'.format(s, attack, i), label)
        image.append(temp_image.copy())
        label.append(temp_label.copy())
        print("adv 个数:", i, len(temp_image))
    image = np.concatenate(image, axis=0)
    label = np.concatenate(label, axis=0)
    np.save('./adv_image/{}_{}_image'.format(s, attack, ), image)
    np.save('./adv_image/{}_{}_label'.format(s, attack, ), label)


if __name__ == '__main__':
    '''
    mnist svhn fashion cifar10 cifar20
    cw fgsm bim jsma
    '''
    data_lst = ['fashion', ]
    attack_lst = ['jsma', "fgsm", "cw", "bim"]

    for dataset, attack in (itertools.product(data_lst, attack_lst)):
        if dataset == "mnist":
            model_path = model_conf.get_model_path(model_conf.mnist, model_conf.LeNet1)
            s = model_conf.mnist + model_conf.LeNet1
            generate_adv_sample(s, model_path, dataset, attack)
            model_path = model_conf.get_model_path(model_conf.mnist, model_conf.LeNet5)
            s = model_conf.mnist + model_conf.LeNet1
            generate_adv_sample(s, model_path, dataset, attack)
        elif dataset == "fashion":
            model_path = model_conf.get_model_path(model_conf.fashion, model_conf.LeNet1)
            s = model_conf.fashion + model_conf.LeNet1
            generate_adv_sample(s, model_path, dataset, attack)
            model_path = model_conf.get_model_path(model_conf.fashion, model_conf.resNet20)
            s = model_conf.fashion + model_conf.resNet20
            generate_adv_sample(s, model_path, dataset, attack)
        elif dataset == "cifar10":
            model_path = model_conf.get_model_path(model_conf.cifar10, model_conf.vgg16)
            s = model_conf.cifar10 + model_conf.vgg16
            generate_adv_sample(s, model_path, dataset, attack)
            model_path = model_conf.get_model_path(model_conf.cifar10, model_conf.resNet20)
            s = model_conf.cifar10 + model_conf.vgg16
            generate_adv_sample(s, model_path, dataset, attack)
        # elif dataset =="svhn":
        else:
            model_path = model_conf.get_model_path(model_conf.svhn, model_conf.vgg16)
            s = model_conf.svhn + model_conf.vgg16
            generate_adv_sample(s, model_path, dataset, attack)
            model_path = model_conf.get_model_path(model_conf.svhn, model_conf.vgg16)
            s = model_conf.svhn + model_conf.vgg16
            generate_adv_sample(s, model_path, dataset, attack)

    # pool = multiprocessing.Pool(processes=1)
    # for dataset, attack in (itertools.product(data_lst, attack_lst)):
    #     if dataset == "mnist":
    #         model_path = model_conf.get_model_path(model_conf.mnist, model_conf.LeNet1)
    #         s = model_conf.mnist + model_conf.LeNet1
    #         pool.apply_async(generate_adv_sample, (s, model_path, dataset, attack))
    #         # model_path = model_conf.get_model_path(model_conf.mnist, model_conf.LeNet5)
    #         # pool.apply_async(generate_adv_sample, (model_path, dataset, attack))
    #     elif dataset == "fashion":
    #         model_path = model_conf.get_model_path(model_conf.fashion, model_conf.LeNet1)
    #         s = model_conf.fashion + model_conf.LeNet1
    #         pool.apply_async(generate_adv_sample, (s, model_path, dataset, attack))
    #
    #         model_path = model_conf.get_model_path(model_conf.fashion, model_conf.resNet20)
    #         s = model_conf.fashion + model_conf.resNet20
    #         pool.apply_async(generate_adv_sample, (s, model_path, dataset, attack))
    #     elif dataset == "cifar10":
    #         model_path = model_conf.get_model_path(model_conf.cifar10, model_conf.vgg16)
    #         s = model_conf.cifar10 + model_conf.vgg16
    #         pool.apply_async(generate_adv_sample, (s, model_path, dataset, attack))
    #         # model_path = model_conf.get_model_path(model_conf.cifar10, model_conf.resNet20)
    #         # pool.apply_async(generate_adv_sample, (model_path, dataset, attack))
    #     # elif dataset =="svhn":
    #     else:
    #         model_path = model_conf.get_model_path(model_conf.svhn, model_conf.vgg16)
    #         s = model_conf.svhn + model_conf.vgg16
    #         pool.apply_async(generate_adv_sample, (s, model_path, dataset, attack))
    #         # model_path = model_conf.get_model_path(model_conf.svhn, model_conf.LeNet5)
    #         # pool.apply_async(generate_adv_sample, (model_path, dataset, attack))
    # pool.close()
    # pool.join()
