
import pickle as pkl
import numpy as np
import os


def stat_seqs(sub_data_folder):
    with open("../datasets/{}/dataset.pkl".format(sub_data_folder), "rb") as fr:
        dataset = pkl.load(fr)
    sbts_len = []
    total = len(dataset)
    for _, nodes, _, _ in dataset:
        sbts_len.append(len(nodes.split(" ")))
    print("mean: ", np.mean(sbts_len))
    print("median: ", np.median(sbts_len))
    counts = np.bincount(sbts_len)
    #返回众数
    print("mode: ", np.argmax(counts))
    # sorted_seqs = sorted(checked, key=lambda t: t[0],reverse=True)
    # print("max: ",sorted_seqs[0]) //4326
    # print("min: ", sorted_seqs[-1]) //11
    count = {"0-100":[], "100-150":[], "150-200":[],"201-400":[], "401-600":[], "601-800":[], "801-1000":[],"1001-1200":[],"1201-1400":[],"1401-1600":[],"1601-1800":[],"1801-":[]}
    for sbt_len in sbts_len:
        if sbt_len >0 and sbt_len <= 100:
            count['0-100'].append(sbt_len)
        elif sbt_len >100 and sbt_len <= 150:
            count['100-150'].append(sbt_len)
        elif sbt_len >150 and sbt_len <=200:
            count['150-200'].append(sbt_len)
        elif sbt_len >= 201 and sbt_len<=400:
            count['201-400'].append(sbt_len)
        elif sbt_len >= 401 and sbt_len <=600:
            count['401-600'].append(sbt_len)
        elif sbt_len >= 601 and sbt_len <=800:
            count['601-800'].append(sbt_len)
        elif sbt_len >= 801 and sbt_len <=1000:
            count['801-1000'].append(sbt_len)
        elif sbt_len >= 1001 and sbt_len <=1200:
            count['1001-1200'].append(sbt_len)
        elif sbt_len >= 1201 and sbt_len <=1400:
            count['1201-1400'].append(sbt_len)
        elif sbt_len >= 1401 and sbt_len <=1600:
            count['1401-1600'].append(sbt_len)
        elif sbt_len >= 1601 and sbt_len <=1800:
            count['1601-1800'].append(sbt_len)
        elif sbt_len >= 1801:
            count['1801-'].append(sbt_len)
    num_list = []
    for key in count.keys():
        num_list.append(len(count[key]))
        print(key+": ", np.sum(num_list)/total)

# 0-200:  0.5988541928985143
# 201-400:  0.8569283555779401
# 401-600:  0.9492193402165702

# 601-800:  0.9720127171996978
# 801-1000:  0.9826586502140519
# 1001-1200:  0.9896468144044321
# 1201-1400:  0.9944913120120876
# 1401-1600:  0.9959361621757744
# 1601-1800:  0.9970001259128682
# 1801-:  1.0

def stat_comm(sub_data_folder):
    with open("../datasets/{}/dataset.pkl".format(sub_data_folder), "rb") as fr:
        dataset = pkl.load(fr)
    sbts_len = []
    total = len(dataset)
    for _, _, _, comm in dataset:
        sbts_len.append(len(comm.split(" ")))
    print("mean: ", np.mean(sbts_len))
    print("median: ", np.median(sbts_len))
    counts = np.bincount(sbts_len)
    #返回众数
    print("mode: ", np.argmax(counts))
    # sorted_comms = sorted(checked, key=lambda t: t[0],reverse=True)
    count = {"0-3":[],"4-10":[],"11-20":[],"21-30":[],"31-40":[], "41-60":[], "61-80":[], "81-100":[],"101-120":[],"121-140":[],"141-160":[],"161-180":[],"181-205":[]}
    for comm in sbts_len:
        if comm >=0 and comm <4:
            count['0-3'].append(comm)
        elif comm >=4 and comm <11:
            count['4-10'].append(comm)
        elif comm >=11 and comm <21:
            count['11-20'].append(comm)
        elif comm >=21 and comm <31:
            count['21-30'].append(comm)
        elif comm >= 31 and comm<41:
            count['31-40'].append(comm)
        elif comm >= 41 and comm <61:
            count['41-60'].append(comm)
        elif comm >= 61 and comm <81:
            count['61-80'].append(comm)
        elif comm >= 81 and comm <101:
            count['81-100'].append(comm)
        elif comm >= 101 and comm <121:
            count['101-120'].append(comm)
        elif comm >= 121 and comm <141:
            count['121-140'].append(comm)
        elif comm >= 141 and comm <161:
            count['141-160'].append(comm)
        elif comm >= 161 and comm <181:
            count['161-180'].append(comm)
        elif comm >= 181:
            count['181-205'].append(comm)
    num_list = []
    for key in count.keys():
        num_list.append(len(count[key]))
        print(key+": ", np.sum(num_list)/347410)
    print("total length: ", len(sbts_len))


def check_dataset():
    with open("../datasets/smart_contracts/comms_4_20/dataset_train_val_test.pkl", "rb") as fr:
        data = pkl.load(fr)
    print("train: ", len(data['train']))
    print("val: ", len(data['val']))
    print("test: ", len(data['test']))

    with open("../datasets/smart_contracts/comms_4_20/dataset_train_val_test_uniq.pkl", "rb") as fr1:
        data1 = pkl.load(fr1)
    print("val1: ", len(data1['val']))
    print("test: ", len(data1['test']))

if __name__ == "__main__":
    # stat_seqs("smart_contracts/comms_4_20")
    # stat_comm("smart_contracts/comms_4_20")
    check_dataset()

"""
这段代码是一个Python脚本，主要用于分析一个数据集中的序列数据。以下是代码的摘要：

1. 导入所需的库：
   - `pickle` 作为 `pkl` 别名，用于读取和写入二进制数据。
   - `numpy` 作为 `np` 别名，用于数值计算。
   - `os` 用于处理文件路径和系统操作。
2. `stat_seqs(sub_data_folder)` 函数：
   - 从指定路径读取名为 "dataset.pkl" 的二进制文件，存储数据集。
   - 统计数据集中每个项目的序列长度，并将这些长度存储在名为 `sbts_len` 的列表中。
   - 打印序列长度的均值、中位数和众数。
   - 使用 `np.bincount` 统计序列长度的分布情况，并打印众数。
3. `stat_comm(sub_data_folder)` 函数：
   - 与 `stat_seqs` 函数类似，但是统计的是与评论（`comm`）相关的序列长度。
4. `check_dataset()` 函数：
   - 从指定路径读取两个不同的二进制文件，分别存储数据集。
   - 打印训练集、验证集和测试集的大小。
   - 打印第二个数据集中的验证集和测试集的大小。
5. 在 `if __name__ == "__main__":` 之后，调用了 `check_dataset()` 函数，以查看数据集的大小。
值得注意的是，代码中存在一些已注释的部分，这些部分在程序运行时是被忽略的。
另外，代码中存在一些可能的错误，比如在 `stat_comm` 函数中，`for comm in sbts_len` 应该是 `for comm in dataset`，
因为需要遍历评论而不是长度列表。同样，在打印 `num_list` 时，可能需要使用 `key` 而不是 `count[key]`。
"""