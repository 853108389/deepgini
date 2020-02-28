import os


def init():
    os.system("python ./init.py")


def run_apfd():
    print("exp ...")
    os.system("python ./exp_apfd/exp_cifar.py")
    os.system("python ./exp_apfd/exp_fashion.py")
    os.system("python ./exp_apfd/exp_mnist.py")
    os.system("python ./exp_apfd/exp_svhn.py")
    os.system("python ./exp_apfd/exp_y_sa.py")
    print("statistic ...")
    os.system("python ./exp_apfd/statistic_apfd_table.py")
    os.system("python ./exp_apfd/statistic_apfd_figure.py")
    print("plot ...")
    os.system("python ./exp_apfd/plot_apfd_figure.py")


def run_retrain():
    print("exp ...")
    os.system("python ./exp_apfd/exp_retrain.py")
    print("statistic ...")
    os.system("python ./exp_apfd/statistic_retrain_table.py")
    print("plot ...")
    os.system("python ./exp_apfd/splot_retrain_figure.py")


if __name__ == '__main__':
    # init
    init()
    # run apfd
    run_apfd()
    # run retain
    run_retrain()