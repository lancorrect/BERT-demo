import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertConfig, BertModel, BertForSequenceClassification


class BertOnly(nn.Module):
    '''
    BERT直接作为一个分类器使用，通过pytorch_transformers中的BertForSequenceClassification实现
    '''

    def __init__(self, args, hidden_size = 256):
        super(BertOnly, self).__init__()
        config = BertConfig.from_pretrained(args.bert_config)  # 初始化bert的各项参数
        # 初始化bert模型，第一个参数是预训练模型名称，第二个参数是训练的参数
        self.bert = BertForSequenceClassification.from_pretrained(args.bert_config, config=config)

    def forward(self, inputs, labels=None):
        '''
        当模型调用时，进行的是此函数
        input: inputs(输入已经encode好的数据，维度为batch_size x encode_size)
               label(标签，维度大小为batch_size x 2)
        output: logits(结果)
        '''
        logits = self.bert(inputs, labels=labels)  # 模型预测，logits包含两个变量，第一个是loss，第二个是返回的结果logits
        return logits


class BertMixed(nn.Module):
    '''
    BERT只作为一个encoder部分而存在，其后还可以自定义很多层用来进行分类
    '''

    def __init__(self, args):
        super(BertMixed, self).__init__()
        config = BertConfig.from_pretrained(args.bert_config)  # 初始化各项参数
        self.bert = BertModel.from_pretrained(args.bert_config, config=config)  # 初始化bert模型

        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 初始化dropout层
        hidden_size = int(args.hidden_size)  # 隐藏层大小，原来是string类型
        # 定义分类器各个层
        layers = [nn.Linear(config.hidden_size, hidden_size),  # 注意维度大小，从bert出来的向量大小为768
                  nn.ReLU(),  # 激活层
                  nn.Linear(hidden_size, 2) if args.binary else 5]  # 线性层，最后的维度必须是2，符合类别数量
        self.classifier = nn.Sequential(*layers)  # 将分类器按顺序迭加起来

    def forward(self, inputs_ids, token_type_ids):
        '''
        input: input_ids(先经历tokenize，再经历将token转为id过程的数据)
               token_type_ids(判断是否是同一个子句的列表)
        output: logits(返回结果，shape为batch_size x 2)
        '''
        outputs = self.bert(inputs_ids, token_type_ids=token_type_ids)
        # shape是(batch_size, sequence_length, hidden_size)，hidden_size=768,它是模型最后一层输出的隐藏状态。
        # 如果我们要进行分类的话可以直接使用下一行的结果，这个是用来得到每个单词经过bert后的表示，可用在之后的图神经网络上
        last_output = outputs[0]
        # shape是(batch_size, hidden_size)，这是序列的第一个token(classification token)的最后一层的隐藏状态，
        # 它是由线性层和Tanh激活函数进一步处理的。（通常用于句子分类，至于是使用这个表示，
        # 还是使用整个输入序列的隐藏状态序列的平均化或池化，视情况而定）
        pooled_output = outputs[1]
        final_outputs = self.dropout(pooled_output)  # dropout
        logits = self.classifier(final_outputs)  # 用分类器进行分类
        return logits