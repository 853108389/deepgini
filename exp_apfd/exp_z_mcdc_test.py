import numpy as np

sc_dif_array = []
vc_dif_array = []
dc_dif_array = []
from collections import defaultdict


def test(test):
    case_dict = defaultdict(set)
    neuron_map_arr = []
    neuron_pair_num = 0
    neuron_num_arr = [3, 5, 2]
    neuron_con_arr = []
    for i in range(len(neuron_num_arr) - 1):
        temp = neuron_num_arr[i] * neuron_num_arr[i + 1]  # 每相邻两层的神经元相乘,计算出连接数,也就是神经元对数
        neuron_pair_num += temp
        neuron_con_arr.append(temp)
        neuron_map_arr.append(np.zeros((neuron_num_arr[i], neuron_num_arr[i + 1])))

    # neuron_activate_test = []  # shape(层数,)  里面的list, shape(测试集个数,神经元个数)
    neuron_activate_test = test
    #################################################################################################
    # 1. 这次我们要在遍历值数组的同时,统计出神经元对与用例的关系
    # neuron_case_con_arr = np.zeros((len(test), len(test))).astype("object")  # 创建数组 shape (用例数,用例数)
    # 2. 每次要遍历两层,为了避免重复计算,先计算出第一层,从第二层再开始遍历
    pre_test = None
    pre_sign = None
    for i in range(len(neuron_activate_test)):  # 遍历值数组
        if i == 0:
            pre_test = neuron_activate_test[0]  #
            pre_sign = np.where(pre_test > 0, 1, 0)  # 大于0的设置为1 # 小于等于0的设置为0
            continue  # 从第二层开始算
        cur_test = neuron_activate_test[i]
        cur_sign = np.where(cur_test > 0, 1, 0)
        sc_arr = []
        vc_arr = []
        dc_arr = []
        #  构造sc值矩阵
        pre_sign_now = pre_sign.copy()
        cur_sign_now = cur_sign.copy()

        for j in range(len(pre_test) - 1):
            pre_sign_row = pre_sign_now[0]  # 为了避免重复,再比较完第一个用例后,将第一个用例删除掉
            cur_sign_row = cur_sign_now[0]
            pre_sign_now = np.delete(pre_sign_now, 0, axis=0)  # 每次删除最上面一行
            cur_sign_now = np.delete(cur_sign_now, 0, axis=0)  # 每次删除最上面一行

            pre_sign_temp = np.abs(pre_sign_now - pre_sign_row)  # 上一层的符号差
            pre_sc_num = np.sum(pre_sign_temp == 1, axis=1)  # 按行求出为1的个数,这个时候就知道了上一层那些用例符合符号变化

            cur_sign_temp = np.abs(cur_sign_now - cur_sign_row)  # 求差,这样0-0 =0  1-1=0 0-1=1 1-0=1 1代表变化 0代表没变化

            case_index = np.argwhere(pre_sc_num == 1).flatten()  # 选出上一层符号变化为1的用例所对应的索引

            # 选取那些行中,值为1的列,记录其索引
            pre_neu_indx_arr = (np.argwhere(pre_sign_temp[case_index])[:, 1])
            # 横坐标代表第x个用例,纵坐标代表第y个神经元

            # 根据上一层中发生变化的用例的下标
            # 寻找出选出本层中的对应用例
            cur_neu_indx_arr = (np.argwhere(cur_sign_temp[case_index]))  # 元组的集合 元组(第x个用例,第y个神经元)

            # 获得其纵坐标也就是本层发生变化了的神经元
            # cur_sc_neuron_index = np.unique(np.argwhere(cur_case > 0)[:, 1])
            # print("用例j和用例np.argwhere(cur_case > 0)[:, 0]满足mc/dc条件,其覆盖的神经元对为:")

            # 计算第几个用例的时候,要把删除的行数加回去
            case_index = case_index + j + 1
            for ix, pre_neu_indx in enumerate(pre_neu_indx_arr):
                cur_neu_indx = cur_neu_indx_arr[cur_neu_indx_arr[:, 0] == ix][:, 1]  # 当前神经元的下标
                if len(cur_neu_indx) != 0:
                    neuron_map_arr[i - 1][pre_neu_indx, cur_neu_indx] = 1  #
                    print(("第{}/{}层: 用例({},{})--->在神经元({}与{})上满足ssc覆盖".format(i - 1, i, j, case_index[ix], pre_neu_indx,
                                                                              cur_neu_indx)))
                    for x in cur_neu_indx:
                        key = None
                        if j < case_index[ix]:
                            key = str(j) + "_" + str(case_index[ix])
                        else:
                            key = str(case_index[ix]) + "_" + str(j)
                        case_dict[key].add((i - 1, pre_neu_indx, x))

        pre_test = cur_test
        pre_sign = cur_sign

    sum = 0
    for l in neuron_map_arr:
        l_sum = np.sum(l == 1)
        sum += l_sum
    print(neuron_map_arr)
    print(case_dict)

    return sum / neuron_pair_num


#
# def test(neuron_activate_test, d_vc, d_dc):
#     for i in range(len(neuron_activate_test)):  # 遍历值数组
#         cur_test = neuron_activate_test[i]  # 当前层的测试集 shape(测试集个数,神经元个数)
#         # 创建三个数组,用来存放三种变化
#         sc_arr = []  # (测试集个数, )   横坐标代表选中了第x个测试集,纵坐标代表第y个测试集与第x个测试集的差异
#         vc_arr = []
#         dc_arr = []
#         #  构造sc值矩阵
#         cur_sign = np.where(cur_test > 0, 1, 0)  # 大于0的设置为1 # 小于等于0的设置为0
#         for j, cur_case in enumerate(cur_test):  # 遍历每层的神经元的测试集 cur_case shape(神经元个数, )
#             # sc
#             cur_sign_temp = np.abs(cur_sign - cur_sign[j])  # 求差,这样0-0 =0  1-1=0 0-1=1 1-0=1 1代表变化 0代表没变化
#             sc_num = np.sum(cur_sign_temp == 1, axis=1)  # 按行求出为1的个数 # shape(测试集个数,)
#             sc_arr.append(sc_num)
#             # vc  使用"绝对"值变化,要求差值大于阈值,但是符号不变
#             cur_vc_temp = np.fabs(cur_test - cur_test[j])  # 求当前测试集差
#             # 1.过滤掉没有符号变化的数
#             cur_sign_vc_mask = np.where(cur_sign_temp > 0, 0, 1)  # 对sc变化矩阵求反 1->0 0->1,即0代表变化 1代表没变化
#             cur_vc_sc_changed = cur_sign_vc_mask * cur_vc_temp  # 乘以距离sc变化矩阵,发生过变化的会变成0,这样有数值的一定是没有发生过sc变化的
#             # 2.求出差值大于某个距离的个数,记作符号变化的个数
#             vc_sum = np.sum(cur_vc_sc_changed >= d_vc, axis=1)
#             vc_arr.append(vc_sum)
#             # dc  要求某层距离函数变化,但某层的符号全不能变化
#             # 1.某层符号不变化,意味着cur_sign_vc_mask矩阵某一行全是1, 1代表没变化,即全都没变化
#             # 按行求最小值,最小值为1则没变化 得到一个列向量,1代表该行没变化, 0代表该行变化了
#             cur_sign_sc_mask = np.min(cur_sign_vc_mask, axis=1, keepdims=True)
#             mr_dc_temp = cur_vc_temp.copy()
#             mr_dc_temp = mr_dc_temp * cur_sign_sc_mask  # 相乘则过滤掉了变化的行
#             x_norm = np.linalg.norm(mr_dc_temp, ord=np.inf, axis=1, keepdims=False)  # 每行求范数 这里使用l无穷范数 就是最大值
#             dc_num = np.where(x_norm > d_dc, 1, 0)  # 大于阈值的设置1,小于的设置为0,  shape(测试集个数,)
#             dc_arr.append(dc_num)
#
#         sc_dif_array.append(sc_arr)
#         vc_dif_array.append(vc_arr)
#         dc_dif_array.append(dc_arr)
#
#


'''
0.3333333333333333
0.4
0.3
0.1
'''

if __name__ == '__main__':
    # 模拟数据
    # 有四个测试用例
    # 模型有两层
    # 第一层有三个神经元
    # 第二层有无个神经元

    a1 = np.array([
        [0, 0.1, 0.2],
        [5.0, 0.1, 3],
        [2.0, -0.4, 0],
        [0, 2.5, 0.2],  # 这里添加第四层用来判断距离变化  超过了阈值
        [0, 2, 0],  # 这里添加第无层用来判断距离变化  没有超过阈值[0, 2, 0.2],
    ])
    a2 = np.array([
        [0, 0.1, 0.2, 4, 5],
        [5.0, 3, 6, 7, -0.2],
        [0.2, 0.1, 0.2, 4, 5],
        [0, 0.11, 0.12, 8, 5],
        [1, 0, 0.2, 4, 0],
    ])

    a3 = np.array([
        [0, 0.1],
        [5.0, 2],
        [0.2, 5],
        [0, 0.11],
        [1, 0, ],
    ])

    neuron_activate_test = [
        a1, a2, a3
    ]

    d_v = 2
    d_d = 2
    res = test(neuron_activate_test)
    print(res)
    #
    # res = ssc()
    # print("ssc")
    # print(res)
    #
    # res = dsc()
    # print("dsc")
    # print(res)
    #
    # res = dvc()
    # print("dvc")
    # print(res)
    #
    # res = svc()
    # print("svc")
    # print(res)
