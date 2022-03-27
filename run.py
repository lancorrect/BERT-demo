#!/usr/bin/env python3

import argparse
from bert_sentiment.train import train
import bert_sentiment.model as model
from bert_sentiment.data import SSTDataset


argp = argparse.ArgumentParser()
argp.add_argument('--model',
                  help='the model you want to train',
                  choices=['bert_only', 'bert_mixed'],
                  default='bert_mixed')  # 挑选模型类别
argp.add_argument('--bert_config',
                  help='Pretrained BERT configuration',
                  default='bert-large-uncased')  # 挑选预训练模型
argp.add_argument('--hidden_size',
                  help='the hidden state size',
                  default=300)  # 选择隐藏层大小
argp.add_argument('--root',
                  help='Use only root nodes of SST',
                  default=True)  # 是否只用每个数据的根节点
argp.add_argument('--binary',
                  help='Use binary labels, ignore neutrals',
                  default=True)  # 标签是否是二分类
argp.add_argument('--save',
                  help='Save model',
                  default=None)  # 是否保存模型
args = argp.parse_args()


def main(args):
    """Train BERT sentiment classifier."""

    # 加载数据集
    trainset = SSTDataset(args, "train", root=args.root, binary=args.binary)
    devset = SSTDataset(args, "dev", root=args.root, binary=args.binary)
    testset = SSTDataset(args, "test", root=args.root, binary=args.binary)

    # 挑选模型
    if args.model == 'bert_only':
        model_our = model.BertOnly(args)
    else:
        model_our = model.BertMixed(args)

    # 训练
    train(args, model_our, trainset, devset, testset, batch_size=8)


if __name__ == "__main__":
    main(args)
