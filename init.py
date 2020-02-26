import os


def init_data():
    tran_path = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    test_path = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    train_data_path = "./data/SVHN_train_32x32.mat"
    test_data_path = "./data/SVHN_test_32x32.mat"
    if not os.path.exists(train_data_path):
        os.system("curl -o {} {}".format(train_data_path, tran_path))
    if not os.path.exists(test_data_path):
        os.system("curl -o {} {}".format(test_data_path, test_path))


def init_model():
    os.system("python ./model_cifar10.py")
    os.system("python ./model_cifar_vgg16.py")
    os.system("python ./model_fashion.py")
    os.system("python ./model_fashion_resNet20.py")
    os.system("python ./model_mnist.py")
    os.system("python ./model_svhn.py")
    os.system("python ./model_svhn_vgg16.py")


def init_dirs():
    mk_path_arr = ["./adv_image", "./data", "./model", "./src"]
    for x in mk_path_arr:
        if not os.path.exists(x):
            os.makedirs(x)


def init_advs():
    os.system("python ./generate_adv.py")


if __name__ == '__main__':
    # init dirs
    init_dirs()
    # download svhn
    init_data()
    # init model
    print("int model speed is slow , you can download the model")
    # init_model()  # speed is slow , you can download the model
    # gen advs
    print("int advs speed is slow , you can download the model")
    # init_advs() # speed is slow , you can download the advs
