from bert_serving.client import BertClient
import numpy as np
import torch
import scipy.sparse as sp
# from get_node import ast_node
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 嵌入AST节点特征向量
def get_embed(ast_file, max_node):
    X = []
    file = open(ast_file, 'r', encoding='utf-8')
    papers = []
    print("读取AST文件成功...")
    print("调用CoderBERT模型嵌入...")
    # bert-serving-start -model_dir D:\BaiduNetdiskWorkspace\uncased_L-12_H-768_A-12 -num_worker=1 -cpu
    for line in file.readlines():
        dic = json.loads(line)
        papers.append(dic)
    for ast in papers:
        val = []
        for b in ast:
            if 'value' in b.keys():
                val.append(b['value'])
            else:
                val.append('')
            ty = [b['type'] for b in ast]
        node = []
        for i in range(0, len(ty)):
            if val[i] != '':
                node.append(ty[i] + '_' +val[i])
            else:
                node.append(ty[i])
        bc = BertClient() # bert-serving-start -model_dir D:\BaiduNetdiskWorkspace\uncased_L-12_H-768_A-12 -num_worker=1 -cpu
            # bert-serving-start -model_dir /hy-tmp/uncased_L-12_H-768_A-12 -num_worker=1



        matrix = bc.encode(node)
        matrix = np.array(matrix)
        matrix = sp.csr_matrix(matrix, dtype=np.float32)
        feature = torch.FloatTensor(np.array(matrix.todense()))
        if feature.size(0) > max_node:
            features = feature[0:max_node]
        else:
            features = torch.zeros(max_node, 768)
            for k in range(feature.size(0)):
                features[k] = feature[k]
        X.append(features)
    print("CodeBERT模型嵌入操作完成...")
    return X
