# BERT-demo

Description
---
It is a simple demo of BERT, which is also my first step to experiment with my idea. I refer to Manish' work, [Fine-grained Sentiment Classification using BERT](https://arxiv.org/abs/1910.03474), and change some codes which could show what I consider about BERT.

The code contains two models:
* The mdoel using BERT for classification
* The model using BERT only for encoder and with other layers, including dropout and classifier, etc.

Usage
---
Experiments can be run using the ```run.py```
```
python3 run.py [OPTIONS]
  Train BERT model
 
 OPTIONS:
  --model          The model you want to train from two kinds of model
  --bert_config    Pretrained BERT configuration
  --hidden_size    The hidden state size
  --root           Use only root nodes of SST
  --binary         Use binary labels, ignore neutrals
  --save           Save the model files after every epoch
```
