import pandas as pd
import numpy as np
import glob
import os
from pandas import MultiIndex
import model_conf


def my_key(s: str):
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

    res = [s.find(x) > -1 for x in ls]
    return res[::-1]


# TODO: 打印retrain表格
def get_retrain_data(copy_res=False, statistic=False):
    # 表格打印
    def statistic_res(name, model_name):
        if not os.path.exists("./res/statistic_res"):
            os.makedirs("./res/statistic_res")
        base_path = './final_exp/res/{}/{}/'.format(name, model_name)
        list1 = os.listdir(base_path)
        df_acc = pd.DataFrame()
        for i, p in enumerate(list1):
            df_res = pd.read_csv("{}{}".format(base_path, p))
            p = os.path.splitext(p)[0]
            # args_arr = p.split("_")
            # method_name = args_arr[-1]
            # metric_name = "_".join(args_arr[:-2])
            # col_name = metric_name + "_" + str(method_name)
            col_name = p
            # print(col_name)
            if i == 0:
                ts = df_res["ts"]
                df_acc["ts"] = ts
            df_acc[col_name] = df_res["acc"]
        df_acc.to_csv("./res/statistic_res/{}_{}_acc.csv".format(name, model_name))
        # df_adv.to_csv("{}{}_adv.csv".format("./model_retrain/res/", model_conf.get_pair_name(name, model_name)))

    if statistic:
        statistic_res(model_conf.mnist, model_conf.LeNet1)
        statistic_res(model_conf.mnist, model_conf.LeNet5)
        statistic_res(model_conf.svhn, model_conf.LeNet5)
        statistic_res(model_conf.svhn, model_conf.vgg16)
        statistic_res(model_conf.fashion, model_conf.LeNet1)
        statistic_res(model_conf.fashion, model_conf.resNet20)
        statistic_res(model_conf.cifar10, model_conf.vgg16)
        statistic_res(model_conf.cifar10, model_conf.resNet20)

    # if copy_res:
    #     base_dir_list = ["./model_retrain/statistic_res", "./model_retrain2/statistic_res"]
    #     obj_dir = "./res/statistic_res"
    #     for base_dir in base_dir_list:
    #         lst = os.listdir(base_dir)
    #         # lst = glob.glob(base_dir + '/*')
    #         for filename in lst:
    #             copyfile(base_dir + "/" + filename, obj_dir + "/" + filename)  # 拷贝文件
    # base_path_arr = ["./model_retrain/statistic_res/", "./model_retrain2/statistic_res/"]
    # model_name_dcit_arr = [model_name_dict1, model_name_dict2]
    writer = pd.ExcelWriter('./res/retrain.xlsx')
    th_per_arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    base_path = "./res/statistic_res"
    lst = glob.glob(base_path + '/*_acc.csv')

    for path in lst:
        # print(lst)
        p = os.path.split(path)[-1]
        p = os.path.splitext(p)[0]
        ds_md_name = "_".join(p.split("_")[:-1])
        print(ds_md_name)
        df = pd.read_csv(path, index_col=0)
        # 获得对应阈值的行
        df = df.loc[df['ts'].isin(th_per_arr)]
        # print(df)
        # 根据行名对数据进行排序
        # 先对行名排序,然后根据顺序创建索引,最后添加一行索引行,然后按照索引行排序,再将索引行删除
        title_row = df.columns.values
        title_row = title_row[1:]  # 先不算ts
        sorted_title_row = sorted(title_row, key=my_key)
        index_arr = [0]  # 把ts放在第一列
        for title in title_row:
            index_arr.append(sorted_title_row.index(title) + 1)  # 排序
        df.loc[len(df)] = index_arr  # 添加索引行
        new_columns = df.columns[df.loc[df.last_valid_index()].argsort()]
        sort_df = df[new_columns]  # 获得排序后df
        sort_df = sort_df.drop(df.tail(1).index)  # 删除最后一行
        sort_df = sort_df.set_index("ts", drop=True)
        new_title = []
        for pp in sort_df.columns.values:
            if pp.find("random") > -1:
                del sort_df[pp]
                continue
            if pp.find("deepgini") > -1:
                l = "DeepGini"
            else:
                temp_arr = pp.split("_")
                cover_method = temp_arr[-1]
                metrics = temp_arr[0]
                if pp.find("lsc") > -1:
                    args = "(1000,100)"
                    if pp.find("ctm") > -1:
                        del sort_df[pp]
                        continue
                elif pp.find("dsc") > -1:
                    args = "(1000,2)"
                    if pp.find("ctm") > -1:
                        del sort_df[pp]
                        continue
                else:
                    print(temp_arr)
                    args = "({})".format(temp_arr[2])
                l = "{}{}-{}".format(metrics.upper(), args, cover_method.upper())
            new_title.append(l)
        sort_df.columns = new_title
        # TODO: create new table
        # print(sort_df)
        # list_sorted = index_arr
        # # 对相关列进行自定义排序
        # df['word'] = df['word'].astype('category').cat.set_categories(list_sorted)
        # # 结果
        # df_sortes = df.sort_values(by=['word'], ascending=True)
        sort_df_reverse = sort_df.T
        # sort_df.T.values()
        sort_df_reverse.to_excel(writer, sheet_name=ds_md_name)
        print("==================")
    writer.save()


if __name__ == '__main__':
    get_retrain_data(statistic=True)
