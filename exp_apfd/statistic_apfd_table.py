import pandas as pd
import numpy as np
import glob
import os
import model_conf
from pandas import MultiIndex
from exp_apfd.statistic_apfd_figure import compute

# 绘制apfd表格
# TODO: apfd函数

ls = [
    "nac_t_0", "nac_t_0.75", "nac",
    "nbc_std_0", "nbc_std_0.5", "nbc_std_1", "nbc",
    "snac_std_0", "snac_std_0.5", "snac_std_1", "snac",
    "tknc_k_1", "tknc_k_2", "tknc_k_3", "tknc",
    "lsc", "LSC", "dsc", "DSC",
    "kmnc",
    # "ssc", "svc", "vsc", "vvc",
    "deep_metric", "deepgini"
]


def my_key(s: str):
    res = [s.find(x) > -1 for x in ls]
    return res[::-1]


# apfd 表格
def get_df(data):
    l = []
    l = l + list(zip(["NAC"] * 2, ["0", "0.75"]))
    l = l + list(zip(["NBC"] * 3, [0, 0.5, 1]))
    l = l + list(zip(["SNAC"] * 3, [0, 0.5, 1]))
    l = l + list(zip(["TKNC"] * 3, [1, 2, 3]))
    l = l + list(zip(["LSC"], ["(1000,100)"]))
    l = l + list(zip(["DSC"], ["(1000, 2)"]))
    l = l + list(zip(["KMNC"] * 2, [1000, 10000]))
    # l = l + list(zip(["SSC"], ["-"]))
    # l = l + list(zip(["SVC"], ["-"]))
    # l = l + list(zip(["VSC"], ["-"]))
    # l = l + list(zip(["VVC"], ["-"]))
    l = l + list(zip(["DeepGini"], ["-"]))

    #  print(l)
    m = MultiIndex.from_tuples(l, names=["Metrics", "Param."])
    # l2 = [("Original Tests", "CTM", "seconds")("Original Tests", "CTM", "APFD")]
    #
    # mm = MultiIndex(levels=[['Original Tests', 'Original Tests ADV'], ["CAM", "CTM"], ["A", "B"]],
    #                 codes=[[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1, ]],
    #                 )
    mm = MultiIndex(
        levels=[['Original Tests', "Original Tests + Adv"], ["Max Coverage (%)", "# Tests to Max Cov.", "CTM", "CAM", ],
                [" ", "seconds", "APFD"]],
        codes=[[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 1, 2, 2, 3, 3, 0, 1, 2, 2, 3, 3],
               [0, 0, 1, 2, 1, 2, 0, 0, 1, 2, 1, 2]],
        # codes=[[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]],
    )
    print(mm)
    df1 = pd.DataFrame(data, index=m, columns=mm)
    # row = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # df1 = df1.append(row, ignore_index=True)
    # print(df1)
    return df1


def get_data(use_time_key=True):
    dir_list = ["output_mnist", "output_cifar", "output_fashion", "output_svhn"]
    # dir_list = ["output_svhn", ]
    base_path = "./all_output"

    writer = pd.ExcelWriter('./result/apfd.xlsx')
    for path in dir_list:
        lst = glob.glob(base_path + '/' + path + '/*')
        # print(lst)
        dataset_name = os.path.basename(path)[7:]
        print(dataset_name)
        for i in lst:  # 遍历每个模型
            rows_arr = []
            model_name = os.path.basename(i)
            adv_data_list = sorted(glob.glob(i + "/*adv*"), key=my_key)
            data_list = sorted(list(set(glob.glob(i + "/*")) - set(adv_data_list)), key=my_key)
            # print(adv_data_list)
            # print(data_list)

            path_arr = list(zip(data_list, adv_data_list))  # 每个模型下所有的文件
            # print(path_arr)

            print(model_name)
            # print(len(path_arr))
            for inx, path in enumerate(path_arr):
                row_arr = []
                # print(path, len(path))
                for key in path:
                    print(key)
                    df = pd.read_csv(key, index_col=0)
                    if "overtime" in list(df) and (df["overtime"] == 1).all():
                        # 超时了
                        row_arr += ["T.O."] * 6
                        continue
                    max_c_per = (100 * df.rate.iloc[0])
                    max_c_num = "N/A"
                    if max_c_per == 0:
                        if key.find("deep_metric") > 0:
                            max_c_per = "N/A"  # 只有deep没有覆盖率
                        else:
                            max_c_per = 0  # 转成int
                            max_c_num = 0
                    else:
                        if max_c_per >= 1:
                            max_c_per = (round(max_c_per))  # 四舍五入
                            max_c_per = int(max_c_per)  # 去掉后面的.0
                        else:
                            max_c_per = '{:.2f}'.format(max_c_per, )
                        max_c_num = (df.cam != 0).sum()
                    print('覆盖率:{}'.format(max_c_per))
                    row_arr.append(max_c_per)
                    # if key.find("lsc") >= 0 or key.find("dsc") >= 0 or key.find("LSC") >= 0 or key.find("DSC") >= 0:
                    #     s = key[key.find("k"):]
                    #     k = int(s.split("_")[1])
                    #     max_c_num = str(int(df.rate.iloc[0] * k))  # lsc,dsc的覆盖率
                    print('覆盖最少样本数:{}'.format(max_c_num))
                    row_arr.append(max_c_num)
                    # and (df["ctm"] == 0).all()
                    if "ctm" in df.columns and "None" not in df["ctm"] and df["ctm"] is not None and not df[
                        "ctm_time"].isnull().all() and key.find("LSC") == -1 and key.find("DSC") == -1:
                        ctm_time = abs(df.ctm_time.iloc[0])  # TODO:
                        if ctm_time < 1:
                            ctm_time = '{:.2f}'.format(ctm_time)
                        else:
                            # ctm_time = int(round(ctm_time))
                            ctm_time = str(round(ctm_time))
                        # ctm_time = "time"
                        res = compute(os.path.basename(key), key, to_csv=False)
                        ctm_apfd = res["ctm"]
                        # print("ctm_apfd=====,", ctm_apfd)
                        ctm_apfd = '{:.3f}'.format(ctm_apfd, )
                    else:
                        ctm_time = "N/A"
                        ctm_apfd = "N/A"
                    print("ctm time {}".format(ctm_time))
                    row_arr.append(ctm_time)
                    print("ctm apfd {}".format(ctm_apfd))
                    row_arr.append(ctm_apfd)
                    # and ( df["cam"] == 0).all()
                    if "cam" in df.columns and "None" not in df["cam"] and df["cam"] is not None and not df[
                        "cam_time"].isnull().all():
                        cam_time = abs(df.cam_time.iloc[0])
                        if cam_time < 1:
                            cam_time = '{:.2f}'.format(cam_time)
                        else:
                            cam_time = str(round(cam_time))
                            # cam_time = int(round(cam_time))
                        # cam_time = "time"
                        res = compute(os.path.basename(key), key, to_csv=False)
                        cam_apfd = res["cam"]
                        cam_apfd = '{:.3f}'.format(cam_apfd)
                    else:
                        cam_time = "N/A"
                        cam_apfd = "N/A"

                    print("cam time {}".format(cam_time))
                    row_arr.append(cam_time)
                    print("cam apfd {}".format(cam_apfd))
                    row_arr.append(cam_apfd)

                    print('错误样本数:{}'.format(pd.value_counts(df.right)[0]))
                    print('总样本数:{}'.format(len(df)))
                    if dataset_name == model_conf.mnist and model_name == model_conf.LeNet5 and key.find("kmnc") > 0:
                        # kmnc 中mnist的LeNet5没有adv
                        row_arr += ["T.O.", "T.O.", "N/A", "N/A", "T.O.", "T.O."]
                        break
                print(row_arr)
                rows_arr.append(row_arr)
                if inx == len(path_arr) - 2:
                    if str(path).find("kmnc") < 0:  # 如果没有kmnc
                        rows_arr.insert(-1, ["T.O.", "T.O.", "N/A", "N/A", "T.O.", "T.O.", "T.O.", "T.O.", "N/A", "N/A",
                                             "T.O.",
                                             "T.O."])  # 添加两行to
                        rows_arr.insert(-1, ["T.O.", "T.O.", "N/A", "N/A", "T.O.", "T.O.", "T.O.", "T.O.", "N/A", "N/A",
                                             "T.O.",
                                             "T.O."])
                    else:
                        # 最后一行添加一行to
                        rows_arr.append(
                            ["T.O.", "T.O.", "N/A", "N/A", "T.O.", "T.O.", "T.O.", "T.O.", "N/A", "N/A", "T.O.",
                             "T.O."])
            table = np.array(rows_arr)
            print(table)
            # print(rows_arr)
            print(table.shape)
            print("==================")
            df = get_df(table)
            df.to_excel(writer, sheet_name=model_conf.get_pair_name(dataset_name, model_name))
            print("==================")
    writer.save()


if __name__ == '__main__':
    get_df(get_data())
