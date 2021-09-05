import argparse
import json
import multiprocessing
import os
from PIL import Image
import random
import re
import pandas as pd
from glob import glob
from importlib import import_module
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import multi_head_attention_forward, sigmoid
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import ValidDataset
from loss import create_criterion


# 학습한 모델을 재생산하기 위해 seed를 고정
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# train하는 동안 learning rate 얻음(공식처럼 사용됨)
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = ValidDataset.decode_multi_class(gt)
        pred_decoded_labels = ValidDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


# 경로
def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


# 이미지 정보 가져옴
def get_img_stats(img_paths):
    img_info = dict(means=[], stds=[])
    for img_path in tqdm(img_paths):
        img = np.array(Image.open(glob(img_path)[0]).convert('RGB'))
        img_info['means'].append(img.mean(axis=(0,1)))
        img_info['stds'].append(img.std(axis=(0,1)))
    return img_info


def get_perfect_df():
    perfect_df = pd.read_csv('/opt/ml/input/data/train/perfect_train.csv')
    perfect_df['people_path'] = perfect_df['path'].apply(lambda x: x.split('/')[-2])
    perfect_df['sub_label'] = perfect_df.apply(lambda x: x['gender'] * 3 + x['age'], axis=1)

    sub_label_df = perfect_df.drop_duplicates(subset=['people_path', 'sub_label']).reset_index(drop=True)

    return perfect_df, sub_label_df


# cv=True일 경우 cross validation 실행
def cross_validation(model_dir, args, k_folds=5):
    seed_everything(args.seed)

    # perfect_df : train 이미지에 대한 정확한 라벨링이 된 data frame
    # sub_label_df : train 2700 명의 대한 age, gender 로만 label 작업 한 data frame
    perfect_df, sub_label_df = get_perfect_df()

    # fold 별 valid_f1 저장할 list
    fold_valid_f1_list = []
    # age, gender 의 비율을 유지시키면서 cross validation 을 구현할 수 있는 라이브러리 StratifiedKFold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=args.seed)
    # train index, valid index 를 뱉어줌
    for n_iter, (train_idx, valid_idx) in enumerate(skf.split(sub_label_df, sub_label_df.sub_label), start=1):
        print(f'>> Cross Validation {n_iter} Starts!')
        train_people_path = sub_label_df.iloc[train_idx].people_path.tolist()
        valid_people_path = sub_label_df.iloc[valid_idx].people_path.tolist()

        train_df = perfect_df[perfect_df['people_path'].isin(train_people_path)]
        valid_df = perfect_df[perfect_df['people_path'].isin(valid_people_path)]

        best_valid_f1 = cv_train(model_dir, args, train_df, valid_df)
        fold_valid_f1_list.append(best_valid_f1)

    print('>> Cross Validation Finish')
    print(f'CV F1-Score: {np.mean(fold_valid_f1_list)}')


def cv_train(model_dir, args, train_df, valid_df):
    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    num_classes = args.num_classes

    # -- dataset
    train_dataset_module = getattr(import_module("dataset"), 'TrainDataset')
    train_dataset = train_dataset_module(
        train_df=train_df,
        features=args.features
    )

    valid_dataset_module = getattr(import_module("dataset"), 'ValidDataset')
    valid_dataset = valid_dataset_module(
        valid_df=valid_df,
        features=args.features
    )

    # -- augmentation
    train_transform_module = getattr(import_module("dataset"), args.augmentation)
    transform = train_transform_module(
        resize=args.resize,
        mean=train_dataset.mean,
        std=train_dataset.std,
    )
    train_dataset.set_transform(transform)

    base_transform_module = getattr(import_module("dataset"), 'BaseAugmentation')
    transform = base_transform_module(
        resize=args.resize,
        mean=train_dataset.mean,
        std=train_dataset.std,
    )
    valid_dataset.set_transform(transform)

    # -- data_loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: Model
    model = model_module(
        model_arch=args.model_name,
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    if args.optimizer == 'MADGRAD':
        opt_module = getattr(import_module("optimizer"), args.optimizer)
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            # weight_decay=5e-4,
        )
    else:
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            # weight_decay=5e-4,
        )
    scheduler = CyclicLR(
        optimizer,
        base_lr=1e-5,
        max_lr=args.lr,
        step_size_down=len(train_dataset) * 2 // args.batch_size,
        step_size_up=len(train_dataset) // args.batch_size,
        cycle_momentum=False,
        mode="triangular2")
    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_train_acc = best_valid_acc = 0
    best_train_loss = best_valid_loss = np.inf
    best_train_f1 = best_valid_f1 = 0
    for epoch in range(args.epochs):
        # train loop
        model.train()
        train_batch_loss = []
        train_batch_accuracy = []
        train_batch_f1 = []
        pbar = tqdm(train_loader)
        for idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            # cutmix 적용 부분
            if np.random.random() <= args.cutmix:
                W = inputs.shape[2]
                mix_ratio = np.random.beta(1, 1)
                cut_W = np.int(W * mix_ratio)
                bbx1 = np.random.randint(W - cut_W)
                bbx2 = bbx1 + cut_W

                rand_index = torch.randperm(len(inputs))
                target_a = labels # 원본 이미지 label
                target_b = labels[rand_index] # 패치 이미지 label

                inputs[:, :, :, bbx1:bbx2] = inputs[rand_index, :, :, bbx1:bbx2]
                outs = model(inputs)
                loss = criterion(outs, target_a) * mix_ratio + criterion(outs, target_b) * (1. - mix_ratio)# 패치 이미지와 원본 이미지의 비율에 맞게 loss 계산
            
            else: # cutmix가 실행되지 않았을 경우
                outs = model(inputs)
                loss = criterion(outs, labels)

            preds = torch.argmax(outs, dim=-1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_batch_loss.append(
                loss.item()
            )
            train_batch_accuracy.append(
                (preds == labels).sum().item() / args.batch_size
            )
            f1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')
            train_batch_f1.append(
                f1
            )

            pbar.set_description(
                f'Epoch #{epoch:2.0f} | '
                f'train | f1 : {train_batch_f1[-1]:.5f} | accuracy : {train_batch_accuracy[-1]:.5f} | '
                f'loss : {train_batch_loss[-1]:.5f} | lr : {get_lr(optimizer):.7f}'
            )

            if (idx + 1) % args.log_interval == 0:
                train_loss = sum(train_batch_loss[idx - args.log_interval:idx]) / args.log_interval
                train_acc = sum(train_batch_accuracy[idx - args.log_interval:idx]) / args.log_interval
                train_f1 = sum(train_batch_f1[idx - args.log_interval:idx]) / args.log_interval

                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/f1-score", train_f1, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/learing_rate", get_lr(optimizer), epoch * len(train_loader) + idx)
            scheduler.step()

        train_item = (sum(train_batch_loss) / len(train_loader),
                      sum(train_batch_accuracy) / len(train_loader),
                      sum(train_batch_f1) / len(train_loader))
        best_train_loss = min(best_train_loss, train_item[0])
        best_train_acc = max(best_train_acc, train_item[1])
        best_train_f1 = max(best_train_f1, train_item[2])

        # val loop
        with torch.no_grad():
            model.eval()
            valid_batch_loss = []
            valid_batch_accuracy = []
            valid_batch_f1 = []
            figure = None
            pbar = tqdm(valid_loader, total=len(valid_loader))
            for (inputs, labels) in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                valid_batch_loss.append(
                    criterion(outs, labels).item()
                )
                valid_batch_accuracy.append(
                    (labels == preds).sum().item() / args.valid_batch_size
                )
                f1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')
                valid_batch_f1.append(
                    f1
                )

                pbar.set_description(
                    f'valid | f1 : {valid_batch_f1[-1]:.5f} | accuracy : {valid_batch_accuracy[-1]:.5f} | '
                    f'loss : {valid_batch_loss[-1]:.5f} | lr : {get_lr(optimizer):.7f}'
                )

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = valid_dataset.denormalize_image(inputs_np, valid_dataset.mean, valid_dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=True
                    )

            valid_item = (sum(valid_batch_loss) / len(valid_loader),
                          sum(valid_batch_accuracy) / len(valid_loader),
                          sum(valid_batch_f1) / len(valid_loader))
            best_valid_loss = min(best_valid_loss, valid_item[0])
            best_valid_acc = max(best_valid_acc, valid_item[1])
            best_valid_f1 = max(best_valid_f1, valid_item[2])
            cur_f1 = valid_item[2]

            if cur_f1 >= 0.7:
                if cur_f1 == best_valid_f1:
                    print(f"New best model for valid f1 : {cur_f1:.5%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best_{cur_f1:.4f}.pth")
                    best_valid_f1 = cur_f1
                else:
                    torch.save(model.module.state_dict(), f"{save_dir}/last_{cur_f1:.4f}.pth")

            print(
                f"[Train] f1 : {train_item[2]:.5}, best f1 : {best_train_f1:.5} || "
                f"acc : {train_item[1]:.5%}, best acc: {best_train_acc:.5%} || "
                f"loss : {train_item[0]:.5}, best loss: {best_train_loss:.5} || "
            )
            print(
                f"[Valid] f1 : {valid_item[2]:.5}, best f1 : {best_valid_f1:.5} || "
                f"acc : {valid_item[1]:.5%}, best acc: {best_valid_acc:.5%} || "
                f"loss : {valid_item[0]:.5}, best loss: {best_valid_loss:.5} || "
            )

            logger.add_scalar("Val/loss", valid_item[0], epoch)
            logger.add_scalar("Val/accuracy", valid_item[1], epoch)
            logger.add_scalar("Val/f1-score", valid_item[2], epoch)
            logger.add_figure("results", figure, epoch)
            print()

    return best_valid_f1


# multi train
def multi_train(model_dir, args):
    features = ['age', 'gender', 'mask']
    criterions = ['cross_entropy', 'cross_entropy', 'cross_entropy']
    classes = [3, 2, 3]

    for feature, criterion, num_classes in zip(features, criterions, classes):
        print(f"-----{feature}-----")
        args.criterion = criterion
        args.num_classes = num_classes
        args.name = args.name+'_'+feature
        args.features = feature
        if args.multi == 1:
            train(model_dir, args)
        elif args.multi == 2:
            cross_validation(model_dir, args, 5)

        args.name = args.name.split('_')[0]


# 학습
def train(model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    num_classes = args.num_classes

    # -- dataset
    # perfect_df : train 이미지에 대한 정확한 라벨링이 된 data frame
    # sub_label_df : train 2700 명의 대한 age, gender 로만 label 작업 한 data frame
    perfect_df, sub_label_df = get_perfect_df()
    # age, gender 를 기준으로 2700명을 8:2 로 train / valid set 으로 분리
    sub_train_df, sub_valid_df = train_test_split(sub_label_df, test_size=0.2,
                                                  stratify=sub_label_df.sub_label, random_state=args.seed)
    train_people_path = sub_train_df.people_path.tolist()
    valid_people_path = sub_valid_df.people_path.tolist()

    train_df = perfect_df[perfect_df['people_path'].isin(train_people_path)]
    valid_df = perfect_df[perfect_df['people_path'].isin(valid_people_path)]

    train_dataset_module = getattr(import_module("dataset"), 'TrainDataset')
    train_dataset = train_dataset_module(
        train_df=train_df,
        features=args.features
    )

    valid_dataset_module = getattr(import_module("dataset"), 'ValidDataset')
    valid_dataset = valid_dataset_module(
        valid_df=valid_df,
        features=args.features
    )

    # -- augmentation
    train_transform_module = getattr(import_module("dataset"), args.augmentation)
    transform = train_transform_module(
        resize=args.resize,
        mean=train_dataset.mean,
        std=train_dataset.std,
    )
    train_dataset.set_transform(transform)

    base_transform_module = getattr(import_module("dataset"), 'BaseAugmentation')
    transform = base_transform_module(
        resize=args.resize,
        mean=train_dataset.mean,
        std=train_dataset.std,
    )
    valid_dataset.set_transform(transform)


    # -- data_loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: Model
    model = model_module(
        model_arch=args.model_name,
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    if args.criterion == 'weight_cross_entropy':
        criterion = create_criterion(args.criterion,
                                     weight=torch.FloatTensor([0.855, 0.892, 0.978, 0.806, 0.784, 0.971,
                                                               0.971, 0.978, 0.996, 0.961, 0.957, 0.994,
                                                               0.971, 0.978, 0.996, 0.961, 0.957, 0.994]).to(device),
                                     reduction='mean')
    else:
        criterion = create_criterion(args.criterion)  # default: cross_entropy
    if args.optimizer == 'MADGRAD':
        opt_module = getattr(import_module("optimizer"), args.optimizer)
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            # weight_decay=5e-4,
        )
    else:
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            # weight_decay=5e-4,
        )
    scheduler = CyclicLR(
        optimizer,
        base_lr=1e-5,
        max_lr=args.lr,
        step_size_down=len(train_dataset) * 2 // args.batch_size,
        step_size_up=len(train_dataset) // args.batch_size,
        cycle_momentum=False,
        mode="triangular2")
    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_train_acc = best_valid_acc = 0
    best_train_loss = best_valid_loss = np.inf
    best_train_f1 = best_valid_f1 = 0
    for epoch in range(args.epochs):
        # train loop
        model.train()
        train_batch_loss = []
        train_batch_accuracy = []
        train_batch_f1 = []
        pbar = tqdm(train_loader)
        for idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            # cutmix 적용 부분
            if np.random.random() <= args.cutmix:
                W = inputs.shape[2]
                mix_ratio = np.random.beta(1, 1)
                cut_W = np.int(W * mix_ratio)
                bbx1 = np.random.randint(W - cut_W)
                bbx2 = bbx1 + cut_W

                rand_index = torch.randperm(len(inputs))
                target_a = labels
                target_b = labels[rand_index]

                inputs[:, :, :, bbx1:bbx2] = inputs[rand_index, :, :, bbx1:bbx2]
                outs = model(inputs)
                loss = criterion(outs, target_a) * mix_ratio + criterion(outs, target_b) * (1. - mix_ratio)
            else:
                outs = model(inputs)
                loss = criterion(outs, labels)

            preds = torch.argmax(outs, dim=-1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_batch_loss.append(
                loss.item()
            )
            train_batch_accuracy.append(
                (preds == labels).sum().item() / args.batch_size
            )
            f1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')
            train_batch_f1.append(
                f1
            )

            pbar.set_description(
                f'Epoch #{epoch:2.0f} | '
                f'train | f1 : {train_batch_f1[-1]:.5f} | accuracy : {train_batch_accuracy[-1]:.5f} | '
                f'loss : {train_batch_loss[-1]:.5f} | lr : {get_lr(optimizer):.7f}'
            )

            if (idx + 1) % args.log_interval == 0:
                train_loss = sum(train_batch_loss[idx-args.log_interval:idx]) / args.log_interval
                train_acc = sum(train_batch_accuracy[idx-args.log_interval:idx]) / args.log_interval
                train_f1 = sum(train_batch_f1[idx-args.log_interval:idx]) / args.log_interval

                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/f1-score", train_f1, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/learing_rate", get_lr(optimizer), epoch * len(train_loader) + idx)
            scheduler.step()

        train_item = (sum(train_batch_loss) / len(train_loader),
                      sum(train_batch_accuracy) / len(train_loader),
                      sum(train_batch_f1) / len(train_loader))
        best_train_loss = min(best_train_loss, train_item[0])
        best_train_acc = max(best_train_acc, train_item[1])
        best_train_f1 = max(best_train_f1, train_item[2])
        
        # val loop
        with torch.no_grad():
            model.eval()
            valid_batch_loss = []
            valid_batch_accuracy = []
            valid_batch_f1 = []
            figure = None
            pbar = tqdm(valid_loader, total=len(valid_loader))
            for (inputs, labels) in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                valid_batch_loss.append(
                    criterion(outs, labels).item()
                )
                valid_batch_accuracy.append(
                    (labels == preds).sum().item() / args.valid_batch_size
                )
                f1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')
                valid_batch_f1.append(
                    f1
                )

                pbar.set_description(
                    f'valid | f1 : {valid_batch_f1[-1]:.5f} | accuracy : {valid_batch_accuracy[-1]:.5f} | '
                    f'loss : {valid_batch_loss[-1]:.5f} | lr : {get_lr(optimizer):.7f}'
                )

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = valid_dataset.denormalize_image(inputs_np, valid_dataset.mean, valid_dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=True
                    )

            valid_item = (sum(valid_batch_loss) / len(valid_loader),
                          sum(valid_batch_accuracy) / len(valid_loader),
                          sum(valid_batch_f1) / len(valid_loader))
            best_valid_loss = min(best_valid_loss, valid_item[0])
            best_valid_acc = max(best_valid_acc, valid_item[1])
            best_valid_f1 = max(best_valid_f1, valid_item[2])
            cur_f1 = valid_item[2]

            if cur_f1 >= 0.7:
                if cur_f1 == best_valid_f1:
                    print(f"New best model for valid f1 : {cur_f1:.5%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best_{cur_f1:.4f}.pth")
                    best_valid_f1 = cur_f1
                else:
                    torch.save(model.module.state_dict(), f"{save_dir}/last_{cur_f1:.4f}.pth")

            print(
                f"[Train] f1 : {train_item[2]:.5}, best f1 : {best_train_f1:.5} || " 
                f"acc : {train_item[1]:.5%}, best acc: {best_train_acc:.5%} || "
                f"loss : {train_item[0]:.5}, best loss: {best_train_loss:.5} || "
            )
            print(
                f"[Valid] f1 : {valid_item[2]:.5}, best f1 : {best_valid_f1:.5} || "
                f"acc : {valid_item[1]:.5%}, best acc: {best_valid_acc:.5%} || "
                f"loss : {valid_item[0]:.5}, best loss: {best_valid_loss:.5} || "
            )

            logger.add_scalar("Val/loss", valid_item[0], epoch)
            logger.add_scalar("Val/accuracy", valid_item[1], epoch)
            logger.add_scalar("Val/f1-score", valid_item[2], epoch)
            logger.add_figure("results", figure, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=2021, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train (default: 21)')
    parser.add_argument('--augmentation', type=str, default='TrainAugmentation', help='data augmentation type (default: CustomAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[280, 210], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=30, help='input batch size for training (default: 30)')
    parser.add_argument('--valid_batch_size', type=int, default=120, help='input batch size for validing (default: 120)')
    parser.add_argument('--model', type=str, default='Model', help='model class (default: BaseModel)')
    parser.add_argument('--model_name', type=str, default='efficientnet_b4', help='what kinds of models (default: efficientnet_b4)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='symmetric', help='criterion type (default: symmetric)')
    parser.add_argument('--cutmix', type=float, default=0.8, help='cutmix ratio (if ratio is 0, not cutmix)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=21, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='experiment', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--cv', type=bool, default=False, help='cross validation (default: False)')
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/faces'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--multi', type=int, default=0, help='model train multiclass by age, gender, mask/ 0 : train, 1 : multi train, 2 : multi train with cv')
    parser.add_argument('--features', default=False, help='given in multi train model')



    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    # cross_validation 사용
    if args.cv:
        cross_validation(model_dir, args, 5)
    # multi label classification 사용 (age, gender, mask)
    elif args.multi:
        multi_train(model_dir, args)
    # basic train
    else:
        train(model_dir, args)
