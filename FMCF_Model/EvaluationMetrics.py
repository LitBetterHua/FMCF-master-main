
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import tensorflow as tf
import numpy as np
from rouge import Rouge

class EvaluationMetrics:

    @staticmethod
    def smoothing1_sentence_bleu(reference, candidate):
        chencherry = SmoothingFunction()
        return sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
    @staticmethod
    def smoothing1_corpus_bleu(references, candidates):
        chencherry = SmoothingFunction()
        return corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)

    @staticmethod
    def rouge(reference, candidate):
        rouge = Rouge()
        return rouge.get_scores(" ".join(candidate), " ".join(reference[0]))[0]['rouge-l']['f']

    @staticmethod
    def meteor(reference, candidate):
        return meteor_score([" ".join(reference[0])], " ".join(candidate))

    @staticmethod
    def remove_pad(tokens_list, end_idx, type):
        outputs = []
        for tokens in tokens_list:
            output = []
            for token in tokens:
                if token == end_idx:
                    break
                else:
                    output.append(str(token))
            assert type == "candidates" or type == "references"
            if type == "candidates":
                outputs.append(output[1:])
            elif type == "references":
                outputs.append([output])
        return outputs

if __name__ == "__main__":
    candi = tf.constant([[1234, 12, 4, 5, 34, 1235, 0, 0], [1234, 22, 41, 35, 12, 1235, 0, 0], [1234, 34, 23, 22, 34, 123, 33, 23]])
    candi = candi.numpy().tolist()
    candi = EvaluationMetrics.remove_pad(candi, 1235, "candidates")
    # print(candi)
    refs = tf.constant([[12, 4, 5, 34, 1235, 0, 0, 0], [22, 41, 34, 12, 1235, 0, 0, 0], [34, 23, 22, 34, 123, 33, 23, 1235]])
    refs = EvaluationMetrics.remove_pad(refs.numpy().tolist(), 1235, "references")
    print("refs: ", refs)
    a = []
    for candidate, ref in zip(candi, refs):
        a.append(EvaluationMetrics.smoothing1_sentence_bleu(ref, candidate))
    print(np.mean(a))
    print(EvaluationMetrics.smoothing1_corpus_bleu(refs, candi))
"""
这段代码定义了一个评估指标类 `EvaluationMetrics`，其中包含了几个计算机器翻译质量评估指标的静态方法：

1. `smoothing1_sentence_bleu(reference, candidate)`: 使用平滑函数 SmoothingFunction.method1 计算句子级别的 BLEU 分数。
2. `smoothing1_corpus_bleu(references, candidates)`: 使用平滑函数 SmoothingFunction.method1 计算语料级别的 BLEU 分数。
3. `rouge(reference, candidate)`: 使用 Rouge 指标计算 ROUGE-L 分数。
4. `meteor(reference, candidate)`: 使用 METEOR 指标计算 METEOR 分数。
5. `remove_pad(tokens_list, end_idx, type)`: 去除填充符号并将序列转换为字符串列表。

在主函数中，通过示例数据进行了测试和演示：
- 一组候选翻译（candi）和参考翻译（refs）被转换成了可以输入评估指标的格式。
- 然后分别计算了每个候选翻译和参考翻译的句子级别 BLEU 分数，并计算所有候选翻译的平均分数。
- 最后，计算了整个语料的 BLEU 分数。

请注意，这段代码依赖于 nltk、tensorflow、numpy 和 rouge 库。要正确运行代码，请确保安装了这些依赖库。
"""
