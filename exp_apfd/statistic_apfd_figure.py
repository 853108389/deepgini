import glob
import os
import csv


# 获得apfd figure 表格,用于图像绘制


class Item:
    def __init__(self, item, header, mode):
        self.item = item
        self.header = header
        self.mode = mode

    def get_order(self):
        order = int(self.item[self.header[self.mode]])
        if order == 0 and self.mode == "cam":
            if "ctm" in self.header:
                order = 500000 + int(self.item[self.header["ctm"]])
            else:
                print("!!!!!! cam has order 0, but the sheet does not have ctm")
                order = 500000
        return order

    def get_best_order(self):
        right = int(self.item[self.header["right"]])
        if right == 1:
            return 1000
        else:
            return 0

    def get_worst_order(self):
        right = int(self.item[self.header["right"]])
        if right == 1:
            return 0
        else:
            return 1000


def get_order(item):
    return item.get_order()


def get_best_order(item):
    return item.get_best_order()


def get_worst_order(item):
    return item.get_worst_order()


metric_index = {
    "nac": 0,
    "nbc": 1,
    "snac": 2,
    "tknc": 3,
    "lsc": 4,
    "dsc": 5,
    "kmnc": 6,
    "deep": 7
}

metric_conf = [
    ["ctm", "cam"],
    ["ctm", "cam"],
    ["ctm", "cam"],
    ["cam"],
    ["cam"],
    ["cam"],
    ["cam"],
    ["cam"]
]


def calc_apfd(items):
    n_tests = len(items)
    sigma_o = 0
    k_mis_tests = 0
    o = 0
    for i in items:
        o = o + 1
        if int(i.item[i.header["right"]]) == 0:
            sigma_o = sigma_o + o
            k_mis_tests = k_mis_tests + 1

    apfd = 1 - (1.0 * sigma_o / (k_mis_tests * n_tests)) + 1.0 / (2 * n_tests)
    return apfd


def best(items):
    items.sort(key=get_best_order)
    return calc_apfd(items)


def worst(items):
    items.sort(key=get_worst_order)
    return calc_apfd(items)


def compute(csvname, abspath, outputdir="", to_csv=False):
    conf = csvname.split("_")
    dataset = conf[0]
    withadv = conf[1] == "adv"
    if withadv:
        metric = conf[2].lower()
        metric_param = "_".join(conf[3:])
    else:
        metric = conf[1].lower()
        metric_param = "_".join(conf[2:])

    print("dataset: " + dataset + "; withadv: " + str(withadv) + "; metric: " + metric + "; param: " + metric_param)

    inputfile = abspath
    sortmodes = metric_conf[metric_index[metric]]
    print(sortmodes)
    for sortmode in sortmodes:
        print(sortmode)
        method = sortmode + "_" + os.path.basename(inputfile)
        outputfile = outputdir + method

        if metric == "kmnc" and sortmode == "cam":
            continue

        items = []
        header_map = {}
        csv_file = csv.reader(open(inputfile, 'r'))
        i = 0
        for line in csv_file:
            if i == 0:
                i += 1
                j = 0
                for x in line:
                    header_map[x] = j
                    j += 1
                if sortmode not in header_map.keys():
                    print(method + " does not have mode " + sortmode)
                    exit(0)
                if "right" not in header_map.keys():
                    print(method + " does not col right")
                    exit(0)
            else:
                items.append(Item(line, header_map, sortmode))

        best_apfd = best(items)
        worst_apfd = worst(items)

        items.sort(key=get_order)
        orig_apfd = calc_apfd(items)

        norm_apfd = (orig_apfd - worst_apfd) / (best_apfd - worst_apfd)

        print("best : " + str(best_apfd))
        print("worst : " + str(worst_apfd))

        print(sortmode + " orig apfd : " + str(orig_apfd))
        print(sortmode + " norm apfd : " + str(norm_apfd))

        if to_csv:
            with open(outputfile, "w") as o:
                o.write(method + "\n")
                sum = 0
                for i in items:
                    if int(i.item[header_map["right"]]) == 0:
                        sum += 1
                    o.write(str(sum) + "\n")
    # return norm_apfd


if __name__ == '__main__':
    input_base_path = "./all_output"
    output_base_path = "./result/apfd_figure_csv"
    dir_list = ["output_mnist", "output_cifar", "output_fashion", "output_svhn"]
    for path_dir in dir_list:
        dataset_name = os.path.basename(path_dir)[7:]
        lst = glob.glob(input_base_path + '/' + path_dir + '/*')
        for inputdir in lst:  # 遍历每个模型
            model_name = os.path.basename(inputdir)
            outputdir = output_base_path + "/" + dataset_name + "/" + model_name + "/"
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            print(inputdir, outputdir)
            for filename in os.listdir(inputdir):
                if filename.endswith(".csv"):
                    abspath = os.path.join(inputdir, filename)
                    print("analyzing " + filename + "...")
                    compute(filename, abspath, outputdir=outputdir, to_csv=True)
                    print("")
