import glob

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import model_conf


def my_key(s: str):
    ls = ["deepgini", "random", "lsc", "LSC", "dsc", "DSC", "nac_t_0", "nac_t_0.75", "nac",
          "kmnc",
          "nbc_std_0", "nbc_std_0.5", "nbc_std_1", "nbc",
          "snac_std_0", "snac_std_0.5", "snac_std_1", "snac",
          "tknc_k_1", "tknc_k_2", "tknc_k_3", "tknc",
          "ssc", "svc", "vsc", "vvc", ]
    res = [s.find(x) > -1 for x in ls]
    return res[::-1]


# 绘制apfd曲线
def plot_curve_with_metric(name, model_name, metrics_arr=None, use_random=False, idx=0,
                           cover="cam_ctm", ):
    def get_ts(n=None, k=10, u=100, ):
        th_per_arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # 创建阈值数组
        if n is None:
            # 返回百分比数值数组
            threshold_arr = th_per_arr
        else:
            # 返回具体的个数
            threshold_arr = [n * x // 100 for x in th_per_arr]
        return threshold_arr

    def metrics_filter_func(path: str):
        flag = False
        for met in metrics_arr:
            if path.find(met) > -1:
                flag = True
                break
        return flag

    color_dict = {}

    dict = {
        "deepgini_cam": "crimson",
        "deepgini2": "darkgreen",
        "deepgini3": "gold",
        "random": "black",
        "lsc_cam": "orange",
        "lsc_ctm": "bisque",
        "dsc_cam": "deeppink",
        "dsc_ctm": "pink",
        "tknc_cam": "Gray",

        "nac_ctm": "Mediumseagreen",
        "nac_cam": "green",
        "nbc_ctm": "MediumPurple",
        "nbc_cam": "purple",
        "snac_ctm": 'royalblue',
        "snac_cam": 'Mediumblue',
        "kmnc_cam": "skyblue"
    }

    for k, v in dict.items():
        color_dict[k] = v
    base_path = './final_exp/res/{}/{}/'.format(name, model_name)
    # base_path = './final_exp/res/{}/'.format(name)
    list1 = os.listdir(base_path)
    list1.sort(key=my_key)
    if metrics_arr is not None:
        list1 = list(filter(metrics_filter_func, list1))  # 选择目标文件
    ts_arr = get_ts()
    plt.xlabel("percentage of test cases")
    plt.ylabel("acc")
    plt.xticks(range(len(ts_arr)), ts_arr)

    for p in list1:
        file_name = os.path.splitext(p)[0]
        df_res = pd.read_csv("{}{}".format(base_path, p))
        args_arr = file_name.split("_")
        c = dict["{}_{}".format(args_arr[0].lower(), args_arr[-1].lower())]
        al = 0.7
        ls = "--"
        if args_arr[0] == "lsc":
            l = "LSC(1000,100)-{}".format(args_arr[-1].upper())
        elif args_arr[0] == "dsc":
            l = "DSC(1000,2)-{}".format(args_arr[-1].upper())
        elif args_arr[0] == "deepgini":
            l = "DeepGini"
        elif args_arr[0] == "random":
            l = "random"
        else:
            l = str(args_arr[0].upper()) + "(" + str(args_arr[2]) + ")" + "-" + str(args_arr[-1].upper())
        if p.endswith("random.csv"):
            ls = "-"
            al = 1
            if use_random:
                plt.plot(range(1, len(df_res["ts"])), df_res["acc"][1:], label=l, alpha=al, color=c, linestyle=ls)
        else:
            if args_arr[-1] == "cam":
                ls = "-."
                if args_arr[0] == "deepgini":
                    ls = "-"
                    al = 1
                if cover.find("cam") > -1:
                    plt.plot(range(1, len(df_res["ts"])), df_res["acc"][1:], label=l, alpha=al, color=c, linestyle=ls)
            elif args_arr[-1] == "ctm":
                ls = "--"
                if cover.find("ctm") > -1:
                    plt.plot(range(1, len(df_res["ts"])), df_res["acc"][1:], label=l, alpha=al, color=c, linestyle=ls)
            else:
                pass

    plt.legend()
    plt.title(model_conf.get_pair_name(name, model_name))
    plt.savefig('./res/fig/{}/{}/{}.pdf'.format(name, model_name, idx), bbox_inches='tight')
    plt.close()


def exec_plot_retrain(name, model_name):
    print("plot {} {}".format(name, model_name))
    output_path = "./res/fig/{}/{}".format(name, model_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    metrics_arr = ["nac_t_0.75_", "nbc_std_0_", "snac_std_0_", "deepgini"]
    plot_curve_with_metric(name, model_name, metrics_arr, use_random=False, idx=1, cover="cam_ctm")

    metrics_arr = ["tknc_k_1_", "lsc", "dsc", "deepgini"]
    plot_curve_with_metric(name, model_name, metrics_arr, use_random=False, idx=2, cover="cam")

    # metrics_arr = ["kmnc_", "deepgini"]
    # plot_curve_with_metric(name, model_name, metrics_arr, use_random=False, idx=3, cover="cam")


if __name__ == '__main__':
    name = model_conf.mnist
    model_name = model_conf.LeNet5
    exec_plot_retrain(name, model_name)
    name = model_conf.mnist
    model_name = model_conf.LeNet1
    exec_plot_retrain(name, model_name)

    name = model_conf.fashion
    model_name = model_conf.resNet20
    exec_plot_retrain(name, model_name)
    name = model_conf.fashion
    model_name = model_conf.LeNet1
    exec_plot_retrain(name, model_name)

    name = model_conf.cifar10
    model_name = model_conf.resNet20
    exec_plot_retrain(name, model_name)
    name = model_conf.cifar10
    model_name = model_conf.vgg16
    exec_plot_retrain(name, model_name)

    name = model_conf.svhn
    model_name = model_conf.LeNet5
    exec_plot_retrain(name, model_name)
    name = model_conf.svhn
    model_name = model_conf.vgg16
    exec_plot_retrain(name, model_name)
