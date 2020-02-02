import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

"""
apfd图像绘制
"""


# 获得图片标签,颜色,渐变,图线
def get_label(p):
    p = os.path.splitext(p)[0]
    l = p
    ls = "-"
    al = 0.7
    c = "black"

    if p.find("deep_metric") > -1:
        l = "DeepGini"
        al = 1
        c = dict["deepgini_cam"]
    else:
        temp_arr = p.split("_")
        cover_method = temp_arr[0]
        dataset_name = temp_arr[1]
        if temp_arr[2] == "adv":
            is_adv = True
            idx = 1
        else:
            is_adv = False
            idx = 0
        metrics = temp_arr[2 + idx]
        if p.find("LSC") > -1:
            args = "(1000,100)"
        elif p.find("DSC") > -1:
            args = "(1000,2)"
        else:
            args = "({})".format(temp_arr[-1])
        l = "{}{}-{}".format(metrics.upper(), args, cover_method.upper())
        if cover_method == "cam":
            ls = "-."
        else:
            ls = "--"
        c = dict["{}_{}".format(metrics.lower(), cover_method)]
    return l, ls, al, c


# 图像绘制
def plot_apfd(data_path, outputdir, use_adv, metrics_arr=None, idx=0,
              cover="all", ):
    def metrics_filter_func(path: str):
        flag = False
        for met in metrics_arr:
            if path.find(met) > -1:
                flag = True
                break
        return flag

    list1 = os.listdir(data_path)
    if use_adv:
        list_file = list(filter(lambda x: x.split("_")[2] == "adv", list1))
    else:
        # print(list1)
        # print("filter adv")
        list_file = list(filter(lambda x: x.split("_")[2] != "adv", list1))
        # print(list_file)
    if cover == "cam":
        list_file = list(filter(lambda x: x.split("_")[0] == "cam", list_file))
    elif cover == "ctm":
        list_file = list(filter(lambda x: x.split("_")[0] == "ctm", list_file))
    else:
        list_file = list_file
    if metrics_arr is not None:
        # 只选出过滤后的文件
        list_file = list(filter(metrics_filter_func, list_file))
    for i, p in enumerate(list_file):
        l, ls, al, c = get_label(p)
        df_res = pd.read_csv("{}{}".format(data_path, p))
        col = df_res.iloc[:, [0]]
        max_num = col.max().values[0]
        col = col.apply(lambda x: x / max_num * 100)
        col2 = np.array(list(range(len(df_res)))) / (len(df_res) - 1) * 100
        plt.plot(col2, col.values, label=l, alpha=al, color=c, linestyle=ls)
    plt.xlabel("Percentage of test case executed")
    plt.ylabel("Percentage of fault detected")
    plt.legend()
    res_path = "{}{}.pdf".format(outputdir, idx)
    plt.savefig(res_path, bbox_inches='tight')
    res_path = "{}{}.png".format(outputdir, idx)
    plt.savefig(res_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # 参数配置
    params = [
        {
            "use_adv": False,  # 是否使用adv
            "metrics_arr": ["nac_t_0.75.csv", "nbc_std_0.csv", "snac_std_0.csv", "deep_metric"],  # 选择绘制的图线
            "cover": "all",  # 选择覆盖的方法
            "idx": 1,  # 图片编号
        },
        {
            "use_adv": False,
            "metrics_arr": ["tknc_k_1.csv", "LSC", "DSC", "deep_metric", "kmnc"],
            "cover": "cam",
            "idx": 2,
        },
        {
            "use_adv": True,
            "metrics_arr": ["nac_t_0.75.csv", "nbc_std_0.csv", "snac_std_0.csv", "deep_metric", ],
            "cover": "all",
            "idx": 3,
        },
        {
            "use_adv": True,
            "metrics_arr": ["tknc_k_1.csv", "LSC", "DSC", "deep_metric", "kmnc"],
            "cover": "cam",
            "idx": 4,
        },

    ]

    input_base_path = "./result/apfd_figure_csv"
    output_base_path = "./result/fig"
    dir_list = ["mnist", "cifar", "fashion", "svhn"]
    # dir_list = ["mnist"]
    for dataset_name in dir_list:
        lst = os.listdir(input_base_path + '/' + dataset_name)
        for model_name in lst:  # 遍历每个模型
            inputdir = input_base_path + '/' + dataset_name + "/" + model_name + "/"
            outputdir = output_base_path + "/" + dataset_name + "/" + model_name + "/"
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            for param in params:
                use_adv, metrics_arr, cover, idx = param["use_adv"], param["metrics_arr"], param["cover"], param[
                    "idx"]
                print("plot {} {} {}".format(dataset_name, model_name, idx))
                plot_apfd(inputdir, outputdir, use_adv, metrics_arr=metrics_arr, idx=idx, cover=cover)
