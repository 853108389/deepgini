image = "image"
label = "label"
mnist = "mnist"
fashion = "fashion"
cifar10 = "cifar"
svhn = "svhn"

LeNet5 = "LeNet5"
LeNet1 = "LeNet1"
resNet20 = "resNet20"
vgg16 = "vgg16"

model_dic = {"mnist_LeNet5": './model/model_mnist.hdf5',
             "mnist_LeNet1": "./model/model_mnist_LeNet1.hdf5",
             "fashion_resNet20": "./model/model_fashion_resNet20.hdf5",
             "fashion_LeNet1": "./model/model_fashion_LeNet1.hdf5",
             "cifar_vgg16": "./model/model_cifar_vgg16.hdf5",
             "cifar_resNet20": "./model/model_cifar10.h5",
             "svhn_vgg16": "./model/model_svhn_vgg16.hdf5",
             "svhn_LeNet5": "./model/model_svhn.hdf5",
             }

trained_dic = {
    "mnist_LeNet5": "./model_retrain/model/mnist/model_mnist.hdf5",
    "mnist_LeNet1": "./model_retrain/model/mnist_LeNet1/model_mnist_LeNet1.hdf5",
    "svhn_LeNet5": "./model_retrain/model/svhn/model_svhn.hdf5",
    "svhn_vgg16": "./model_retrain/model/svhn/model_svhn_vgg16.hdf5",
    "fashion_resNet20": "./model_retrain/model/fashion/model_fashion_resNet20.hdf5",
    "fashion_LeNet1": "./model_retrain/model/fashion/model_fashion_LeNet1.hdf5",
    "cifar_resNet20": './model_retrain/model/cifar/model_cifar10.h5',
    "cifar_vgg16": "./model_retrain/model/cifar/model_cifar_vgg16.hdf5",
}

# retrained_th_dic = {
#     "mnist_LeNet5": "./model_retrain/model/mnist/{}/model_mnist_ts_{}.hdf5",
#     "mnist_LeNet1": "./model_retrain/model/mnist/{}/model_mnist_LeNet1_ts_{}.hdf5",
#
#     "fashion_resNet20": "./model_retrain/model/fashion/{}/model_fashion_resNet20_ts_{}.hdf5",
#     "fashion_LeNet1": "./model_retrain/model/fashion/{}/model_fashion_LeNet1_ts_{}.hdf5",
#
#     "svhn_LeNet5": './model_retrain/model/svhn/{}/model_svhn_ts_{}.hdf5',
#     "svhn_vgg16": './model_retrain/model/svhn/{}/model_svhn_vgg16_ts_{}.hdf5',
#
#     "cifar_resNet20": './model_retrain/model/cifar/{}/model_cifar_ts_{}.hdf5',
#     "cifar_vgg16": './model_retrain/model/cifar/{}/model_cifar_vgg16_ts_{}.hdf5',
# }

name_list = [mnist, fashion, svhn, cifar10]
model_data = {
    mnist: [LeNet5, LeNet1],
    fashion: [LeNet1, resNet20],
    cifar10: [vgg16, resNet20],
    svhn: [LeNet5, vgg16]
}

pair_list = ["mnist_LeNet5", "mnist_LeNet1", "fashion_resNet20", "fashion_LeNet1", "svhn_LeNet5", "svhn_vgg16",
             "cifar_resNet20", "cifar_vgg16"]


def get_model_path(datasets, model_name):
    return model_dic[datasets + "_" + model_name]


def get_trained_model_path(datasets, model_name):
    return trained_dic[datasets + "_" + model_name]


def get_pair_name(datasets, model_name):
    return datasets + "_" + model_name


def get_retrain_adv_path(attack, dataset, model_name):
    print(dataset + "_" + model_name)
    dir_path = "adv_image2"
    dic = {"mnist_LeNet5": _get_old_adv_path(dir_path, attack, dataset),
           "mnist_LeNet1": _get_new_adv_path(dir_path, attack, dataset, model_name),

           "fashion_resNet20": _get_new_adv_path(dir_path, attack, dataset, model_name),
           "fashion_LeNet1": _get_new_adv_path(dir_path, attack, dataset, model_name),

           "cifar_resNet20": _get_old_adv_path(dir_path, attack, dataset),
           "cifar_vgg16": _get_new_adv_path(dir_path, attack, dataset, model_name),

           "svhn_LeNet5": _get_old_adv_path(dir_path, attack, dataset),
           "svhn_vgg16": _get_new_adv_path(dir_path, attack, dataset, model_name),
           }
    print(dic[dataset + "_" + model_name])
    return dic[dataset + "_" + model_name]


def get_adv_path(attack, dataset, model_name):
    print(dataset + "_" + model_name)
    dir_path = "adv_image"
    dic = {"mnist_LeNet5": _get_old_adv_path(dir_path, attack, dataset),
           "mnist_LeNet1": _get_new_adv_path(dir_path, attack, dataset, model_name),

           "fashion_resNet20": _get_new_adv_path(dir_path, attack, dataset, model_name),
           "fashion_LeNet1": _get_new_adv_path(dir_path, attack, dataset, model_name),

           "cifar_resNet20": _get_old_adv_path(dir_path, attack, dataset),
           "cifar_vgg16": _get_new_adv_path(dir_path, attack, dataset, model_name),

           "svhn_LeNet5": _get_old_adv_path(dir_path, attack, dataset),
           "svhn_vgg16": _get_new_adv_path(dir_path, attack, dataset, model_name),
           }
    print(dic[dataset + "_" + model_name])
    return dic[dataset + "_" + model_name]


def _get_old_adv_path(dir_path, attack, dataset):
    if dataset == "cifar":
        dataset = "cifar10"
    # "./adv_image/bim_cifar10_label.npy"
    i = './{}/{}_{}_{}.npy'.format(dir_path, attack, dataset, image)
    l = './{}/{}_{}_{}.npy'.format(dir_path, attack, dataset, label)
    return i, l


def _get_new_adv_path(dir_path, attack, dataset, model_name):
    # ./adv_image/fashionLeNet1_cw_image.npy
    i = './{}/{}_{}_{}.npy'.format(dir_path, dataset + model_name, attack, image)
    l = './{}/{}_{}_{}.npy'.format(dir_path, dataset + model_name, attack, label)
    return i, l

# return './adv_image/{}_{}_{}_image'.format(model_path, attack, dataset)
