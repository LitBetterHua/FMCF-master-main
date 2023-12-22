
from html.parser import HTMLParser
import networkx as nx
import numpy as np
from data_process.utils import re_0001_, re_0002, re_opt


class MyHTMLParser(HTMLParser):
    def __init__(self):
        super(MyHTMLParser, self).__init__()
        self.parentstack = list()
        self.curtag = -1
        self.tagidx = -1
        self.graph = nx.Graph()
        self.seq = list()

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        self.parentstack.append(self.curtag)
        self.tagidx += 1
        self.seq.append(tag)
        self.graph.add_node(self.tagidx, text=tag)
        if self.parentstack[-1] >= 0:
            self.graph.add_edge(self.parentstack[-1], self.tagidx)
        self.curtag = self.tagidx

    def handle_endtag(self, tag):
        self.curtag = self.parentstack.pop()


    def handle_data(self, data):
        # first, do data text preprocessing
        if re_opt.fullmatch(data) is None and data != "$NUM$" and data != "$STR$" and data != "$ADDR$":
            data = re_0001_.sub(re_0002, data).strip()
        if data == "$NUM$" or data == "$STR$" or data == "$ADDR$":
            data = "「" + data[1:-1] + "」"
        else:
            data = data.lower()
        # second, create a node if there is text
        if(data != ''):
            for d in data.split(' '): # each word gets its own node
                if d != '':
                    self.parentstack.append(self.curtag)
                    self.tagidx += 1
                    self.seq.append(d)
                    self.graph.add_node(self.tagidx, text=d)
                    self.graph.add_edge(self.parentstack[-1], self.tagidx)
                    self.curtag = self.tagidx
                    self.curtag = self.parentstack.pop()

    def get_graph(self):
        return(self.graph)

    def get_seq(self):
        return(self.seq)

def xmldecode(unit):
    parser = MyHTMLParser()
    parser.feed(unit)
    return(parser.get_graph(), parser.get_seq())

def xml_graph(contract_folder, xml_file):
    with open("../contracts/contracts_seqs_xml/{}/{}".format(contract_folder, xml_file), "r", encoding="utf-8") as fr:
    # with open("../contracts/test_xml.txt", "r", encoding="utf-8") as fr:
        text = ""
        line = fr.readline()
        while line:
            text += line
            line = fr.readline()
        (graph, _) = xmldecode(text)
        try:
            nodes = list(graph.nodes.data())
            edges = nx.adjacency_matrix(graph)
        except:
            eg = nx.Graph()
            eg.add_node(0)
            nodes = np.asarray([0])
            edges = nx.adjacency_matrix(eg)
        nodes = " ".join([node[1]['text'] for node in nodes])
        return nodes, edges

if __name__ == "__main__":
    nodes, edges = xml_graph("contract1", "44_35.txt")
    print(nodes)
    print(edges.todense())

    """
    这段代码实现了一个 HTML 解析器（`MyHTMLParser`），用于解析 HTML 标记并构建相关的图形结构。下面是代码的主要组成部分：

1. 导入必要的库：
   - `HTMLParser`：HTML 解析器库。
   - `networkx`：用于图形处理的库。
   - `numpy`：用于数值操作的库。
   - `re_0001_`、`re_0002`、`re_opt`：从 `data_process.utils` 导入的正则表达式和其他实用函数。

2. 定义 `MyHTMLParser` 类：这是一个自定义的 HTML 解析器类，继承自 `HTMLParser`。它包含以下主要方法：

   - `__init__` 方法：初始化解析器对象，设置初始状态和属性。
   - `handle_starttag` 方法：处理 HTML 开始标记。将标记转换为小写，添加到标记序列中，并在图形中创建相应的节点和边。
   - `handle_endtag` 方法：处理 HTML 结束标记。在图形中更新当前标记。
   - `handle_data` 方法：处理标记之间的文本数据。对文本数据进行预处理（例如，应用正则表达式替换），然后创建相应的节点和边。

   此外，还有两个方法用于获取生成的图形和标记序列。

3. `xmldecode` 函数：该函数接受一个 HTML 单元并使用 `MyHTMLParser` 解析它，然后返回解析得到的图形和标记序列。

4. `xml_graph` 函数：此函数接受一个合同文件夹和一个 XML 文件，从文件中读取文本，然后使用 `xmldecode` 函数解析它。最终，该函数返回标记序列和图形的节点和边的表示。

5. 主程序：在 `if __name__ == "__main__":` 语句块中，代码示例通过调用 `xml_graph` 函数解析合同文件夹中的特定 XML 文件，并打印标记序列和图形的邻接矩阵表示。

总之，这段代码实现了一个用于解析 HTML 标记并构建相关图形的解析器。它可以用于处理包含 HTML 结构的文本数据，将其转化为图形数据，以便进行进一步的分析和处理。
    """


