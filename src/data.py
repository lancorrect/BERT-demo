"""This module defines a configurable SSTDataset class."""

import pytreebank
import torch
from loguru import logger
from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset

logger.info("Loading the tokenizer")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")  # 引入bert-large-base模型

logger.info("Loading SST")
sst = pytreebank.load_sst()  # 倒入数据集，每个数据是记录了一个树状结构，我们只需要提取每个数据的第一个元素就是整个句子


def rpad(array, n=70):
    """Right padding."""
    '''
    填充函数，目的是将所有所有句子的序列长度都相同
    input: array(输入的序列), n(目标长度)
    output:array(长度已经达到目标长度的序列)
    '''
    current_len = len(array)
    if current_len > n:
        return array[: n - 1]
    extra = n - current_len
    return array + ([0] * extra)


def get_binary_label(label):
    """Convert fine-grained label to binary label."""
    '''
    转化标签函数，由于sst数据集中的标签总共分为5类: 0-very negative, 1-negative, 2-neural,
    3-positive, 4-very positive。所以除了neural类别的数据，其他的统统分为两类：0-negtive, 1-positive
    input:label(输入的标签，数据集中自带的标签，为int形式)
    output:0 or 1(返回重新制作的标签)
    '''
    if label < 2:
        return 0
    if label > 2:
        return 1
    raise ValueError("Invalid label")


class SSTDataset(Dataset):
    """Configurable SST Dataset.
    
    Things we can configure:
        - split (train / val / test)
        - root / all nodes
        - binary / fine-grained
    """

    def __init__(self, args, split="train", root=True, binary=True):
        """Initializes the dataset with given configuration.

        Args:
            split: str
                Dataset split, one of [train, val, test]
            root: bool
                If true, only use root nodes. Else, use all nodes.
            binary: bool
                If true, use binary labels. Else, use fine-grained.
        """

        self.args = args

        logger.info(f"Loading SST {split} set")
        self.sst = sst[split]  # 提取训练集、验证集和测试集

        logger.info("Tokenizing")
        # 如果是二分类问题，且是对整个句子判断情感极性
        if root and binary:
            self.data = []  # 初始化数据列表，至于初始化成列表的原因在__getitem__函数中有大作用
            # 遍历数据集
            for tree in self.sst:
                if tree.label == 2:
                    continue  # 如果数据的标签是2,证明是neural的数据，直接跳过
                tokens = []  # 初始化tokenize后存放的序列
                for word in tree.to_lines()[0]:
                    word_token = tokenizer.tokenize(word)  # 对句子中的每个单词进行tokenize
                    tokens.extend(word_token)  # 将结果放入到tokens中，这里的extend里的内容必须是一个可迭代的对象
                tokens = ['CLS']+tokens+['SEP']  # bert的独有形式，cls代表一个分类标志符，sep代表一个子句的结尾
                # 将token转换为id形式，并且要填充好，这是为了在之后将bert只作为encoder部分的模型提供输入
                tokens = tokenizer.convert_tokens_to_ids(rpad(tokens, n=300))
                # 构建分割器，其实就是如果是一个句子的单词，其位置上都是相同的数字，这么做的目的是为了语句配对任务
                # 同样是为了为了之后将bert只作为encoder部分的模型提供输入
                segment_ids = [0]*len(tokens)
                # 将语句中的所有单词encode，这是为了给将bert作为分类模型提供输入
                tokens_encode = rpad(tokenizer.encode('[CLS]'+tree.to_lines()[0]+'[SEP]'), n=66)
                self.data.append({'input_ids': tokens,
                                  'token_type_ids': segment_ids,
                                  'tokens_encode': tokens_encode,
                                  'labels': get_binary_label(tree.label)})
            '''self.data = [
                (
                    rpad(
                        tokenizer.encode("[CLS] " + tree.to_lines()[0] + " [SEP]"), n=66
                    ),
                    get_binary_label(tree.label),
                )
                for tree in self.sst
                if tree.label != 2
            ]'''
        elif root and not binary:
            self.data = [
                (
                    rpad(
                        tokenizer.encode("[CLS] " + tree.to_lines()[0] + " [SEP]"), n=66
                    ),
                    tree.label,
                )
                for tree in self.sst
            ]
        elif not root and not binary:
            self.data = [
                (rpad(tokenizer.encode("[CLS] " + line + " [SEP]"), n=66), label)
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
            ]
        else:
            self.data = [
                (
                    rpad(tokenizer.encode("[CLS] " + line + " [SEP]"), n=66),
                    get_binary_label(label),
                )
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
                if label != 2
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
        此函数的目的是当调用该类的对象时，如果该对象其后跟着索引，那么将会调用该函数，索引会被作为参数传入进来
        该函数用处在调用torch中的Dataloader函数时，它会根据batch_size抽取一定量的数据，需要用到索引，
        此时该函数就派上用场了，自己想要给Dataloader传入什么数据在这里可以自己定义
        但是一定要注意的是，类中的数据集格式一定要是列表，如果是字典的话就会报错！！！！
        input: index(传入的索引)
        output：x, y
        '''
        elements = self.data[index]  # 得到该索引下的数据
        # 如果是将bert作为分类器的模型，则挑选encode好的token和label，并只将token进行tensor化
        if self.args.model == 'bert_only':
            x = elements['tokens_encode']
            y = elements['labels']
            x = torch.tensor(x)
        else:
            # 如果是只将bert作为模型中的encoder部分，则需要将input_ids和token_type_ids分别进行tensor化
            x = elements['input_ids'], elements['token_type_ids']
            x = tuple(torch.tensor(t) for t in x)
            y = elements['labels']
        return x, y
