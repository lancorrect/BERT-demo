import torch
from loguru import logger
from tqdm import tqdm

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"  # gpu加速


def train_one_epoch(args, model, lossfn, optimizer, dataset, batch_size=32):
    '''
    此函数是训练一代时所用的函数，模型的训练、优化器的优化以及损失的计算和反向传播都在此进行
    input: args(携带各种参数的命令行模块)
           model(模型)
           lossfn(损失函数)
           optimizer(优化器)
           dataset(数据集)
           batch_size(每个batch的大小)
    output: train_loss(训练损失)
            train_acc(训练准确率)
    '''
    # 使用Dataloader模块加载数据
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    train_loss, train_acc = 0.0, 0.0
    # 善用tqdm模块，它会显示模型的训练进度，不需要再像tensorflow一样调用函数才能看见
    for batch, labels in tqdm(generator):
        # 这里的batch有两种情况：
        # 首先是将bert作为分类器的模型情况，batch中只包含token_encode，所以直接用GPU加速即可
        # 其次是将bert作为encoder部分的模型情况，batch中包含input_ids和token_type_ids，需要分别放到GPU上
        batch = tuple(t.to(device) for t in batch) if args.model == 'bert_mixed' else batch.to(device)
        labels = labels.to(device)  # 标签同样需要放到GPU上
        optimizer.zero_grad()
        # loss, logits = model(batch, labels=labels)
        if args.model == 'bert_only':
            logits = model(batch, labels=labels)[1]  # 返回的结果包括loss和logits，在这里只需要logits
        else:
            logits = model(batch[0], batch[1])  # 返回的结果只有logits
        err = lossfn(logits, labels)  # 计算损失
        err.backward()  # 反向传播
        optimizer.step()

        train_loss += err.item()  # 计算一代的训练总损失
        pred_labels = torch.argmax(logits, axis=1)  # 挑选概率最大值，并返回位置索引
        train_acc += (pred_labels == labels).sum().item()  # 计算正确的个数
    train_loss /= len(dataset)  # 计算训练损失
    train_acc /= len(dataset)  # 计算训练正确率
    return train_loss, train_acc


def evaluate_one_epoch(args, model, lossfn, optimizer, dataset, batch_size=32):
    '''
    此函数是评价一代时所用的函数
    input: args(携带各种参数的命令行模块)
           model(模型)
           lossfn(损失函数)
           optimizer(优化器)
           dataset(数据集)
           batch_size(每一个batch的大小)
    output: loss(损失)
            acc(正确率)
    '''
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.eval()
    loss, acc = 0.0, 0.0
    with torch.no_grad():
        for batch, labels in tqdm(generator):
            batch = tuple(t.to(device) for t in batch) if args.model == 'bert_mixed' else batch.to(device)
            labels = labels.to(device)
            logits = model(batch)[0] if args.model == 'bert_only' else model(batch[0], batch[1])
            error = lossfn(logits, labels)
            loss += error.item()
            pred_labels = torch.argmax(logits, axis=1)
            acc += (pred_labels == labels).sum().item()
    loss /= len(dataset)
    acc /= len(dataset)
    return loss, acc


def train(args, model, trainset, devset, testset, epochs=30, batch_size=32, save=False):
    '''
    训练函数
    input: args(携带各种参数的命令行模块)
           model(模型)
           trainset(训练集)
           devset(验证集)
           testset(测试集)
           epochs(训练的代数)
           batch_size(每个batch的大小)
           save(是否保存)
    '''

    model = model.to(device)  # GPU加速
    lossfn = torch.nn.CrossEntropyLoss()  # 初始化损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 初始化优化器

    # 按照代数进行迭代，逻辑是训练一代，验证一代并测试一代
    for epoch in range(1, epochs):
        train_loss, train_acc = train_one_epoch(
            args, model, lossfn, optimizer, trainset, batch_size=batch_size
        )  # 训练一代
        val_loss, val_acc = evaluate_one_epoch(
            args, model, lossfn, optimizer, devset, batch_size=batch_size
        )  # 验证一代
        test_loss, test_acc = evaluate_one_epoch(
            args, model, lossfn, optimizer, testset, batch_size=batch_size
        )  # 测试一代
        logger.info(f"epoch={epoch}")  # 打印是第几代
        logger.info(
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}"
        )  # 打印损失信息
        logger.info(
            f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, test_acc={test_acc:.3f}"
        )  # 打印正确率信息
        '''
        if save:
            label = "binary" if binary else "fine"
            nodes = "root" if root else "all"
            torch.save(model, f"{bert}__{nodes}__{label}__e{epoch}.pickle")'''

    logger.success("Done!")  # 训练完成
