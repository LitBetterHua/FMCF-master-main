
import pickle as pkl
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if __package__ is None:
    from Data_Split_Process import Token_Utils
else:
    import Token_Utils


def create_dicts(dataset, sub_data_folder, num_words=30000):
    """
    create dictionaries for seqs and comms.
    :param dataset: dataset_train_val_test.pkl
    :param sub_data_folder: each model has his onw sub_data_folder
    :param num_words: max num of words in the dictionary
    :return:
    """
    sbts, nodes, _, comms = Utils.split_sbts_nodes_comms(dataset['train'])
    print("sbt[0]: ", sbts[0])
    print("nodes[0]: ", nodes[0])
    print("comms[0]: ", comms[0])
    Utils.create_dict(sub_data_folder, "sbts", sbts, num_words)
    Utils.create_dict(sub_data_folder, "nodes", nodes, num_words)
    Utils.create_dict(sub_data_folder, "comms", comms, num_words)


def generate_sequences(dataset, sub_data_folder):
    for type in ['train', 'val', 'test']:
        sbts, nodes, edges, comms = Utils.split_sbts_nodes_comms(dataset[type])
        sbts = Utils.text_to_idx(sub_data_folder, "sbts", sbts)
        nodes = Utils.text_to_idx(sub_data_folder, "nodes", nodes)
        comms = Utils.text_to_idx(sub_data_folder, "comms", comms)
        with open("../datasets/{}/tokens_idx/sbts_{}.pkl".format(sub_data_folder, type),"wb") as fw:
            pkl.dump(sbts, fw)
        with open("../datasets/{}/tokens_idx/nodes_{}.pkl".format(sub_data_folder, type),"wb") as fw:
            pkl.dump(nodes, fw)
        with open("../datasets/{}/tokens_idx/edges_{}.pkl".format(sub_data_folder, type),"wb") as fw:
            pkl.dump(edges, fw)
        with open("../datasets/{}/tokens_idx/comms_{}.pkl".format(sub_data_folder, type),"wb") as fw:
            pkl.dump(comms, fw)

def get_dicts(sub_data_folder):
    parent_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    with open(parent_path + "/datasets/{}/dictionaries/sbts_dic.pkl".format(sub_data_folder), "rb") as fr:
        sbts_dic = pkl.load(fr)
    with open(parent_path + "/datasets/{}/dictionaries/nodes_dic.pkl".format(sub_data_folder), "rb") as fr:
        nodes_dic = pkl.load(fr)
    with open(parent_path + "/datasets/{}/dictionaries/comms_dic.pkl".format(sub_data_folder), "rb") as fr:
        comms_dic = pkl.load(fr)
    return sbts_dic, nodes_dic, comms_dic


if __name__ == "__main__":
    SUB_DATA_FOLDER = "smart_contracts/comms_4_20"
    with open("../datasets/{}/dataset_train_val_test_uniq.pkl".format(SUB_DATA_FOLDER), "rb") as fr:
        dataset = pkl.load(fr)
    # create_dicts(dataset, SUB_DATA_FOLDER)
    # generate_sequences(dataset, SUB_DATA_FOLDER)
    sbts_dic, nodes_dic, comms_dic = get_dicts(SUB_DATA_FOLDER)
    print(len(sbts_dic.word_index))
    print(len(nodes_dic.word_index))
    print(len(comms_dic.word_index))

"""
这段代码的主要目的是处理文本数据，创建字典并生成序列数据。以下是代码的摘要：

1. 导入必要的库和模块：
   - `pickle as pkl`：用于序列化和反序列化数据的模块。
   - `sys` 和 `os`：用于处理文件路径和导入自定义模块。
   - `Utils` 模块：包含了用于数据处理的实用函数。

2. 定义 `create_dicts` 函数：该函数接受一个数据集、子数据文件夹和一个可选的最大字典词汇数。它的主要任务是创建用于序列和评论的字典。具体步骤包括：
   - 调用 `Utils.split_sbts_nodes_comms` 函数，将训练集中的数据拆分为 `sbts`、`nodes`、`edges` 和 `comms` 四个部分。
   - 打印一些示例数据，以便查看数据的结构。
   - 分别调用 `Utils.create_dict` 函数创建 `sbts`、`nodes` 和 `comms` 的字典，可以限制最大字典词汇数。

3. 定义 `generate_sequences` 函数：该函数用于生成序列数据，并将结果保存为 Pickle 文件。具体步骤包括：
   - 针对每个数据集类型（'train'、'val'、'test'）：
     - 调用 `Utils.split_sbts_nodes_comms` 函数，拆分数据集。
     - 使用之前创建的字典将文本转换为索引。
     - 使用 `pickle` 将处理后的序列数据保存为文件。

4. 定义 `get_dicts` 函数：该函数用于获取之前创建的字典。它根据子数据文件夹的名称构建字典文件路径，并加载这些字典。

5. 主程序：在 `if __name__ == "__main__":` 语句块中，执行以下主要步骤：
   - 定义子数据文件夹名称 `SUB_DATA_FOLDER`。
   - 使用 `pickle` 从文件加载数据集，并存储在 `dataset` 变量中。
   - 调用 `get_dicts` 函数获取 `sbts_dic`、`nodes_dic` 和 `comms_dic` 字典，并打印它们的词汇数量。

总之，这段代码用于处理文本数据，创建字典，并将文本数据转换为序列数据。它可以用于文本数据的预处理和序列化，以供后续的机器学习或深度学习任务使用。在主程序中，你可以选择是否执行创建字典和生成序列数据的操作。
"""

