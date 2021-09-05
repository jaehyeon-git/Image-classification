import argparse
import os
import glob
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import TestDataset


# load train model from .pth file
def load_model(model_name, pth_name, saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        model_arch=model_name,
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, pth_name)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


# load cv(multi-labeled) train model from .pth file
def cv_load_model(model_name, pth_name, saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        model_arch=model_name,
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)
    best_cktp_list = sorted(glob.glob(os.path.join(saved_model, pth_name) + '*'))
    if len(best_cktp_list) == 0:
        return
    model_path = best_cktp_list[-1]
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


# inference loaded train model
@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18
    model = load_model(args.model_name, args.pth_name, model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'faces')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    test_dataset = TestDataset(img_paths, args.resize)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for idx, images in enumerate(pbar):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output_{args.pth_name}.csv'), index=False)
    print(f'Inference Done!')


# inference loaded multi-labeled (cv)train models
@torch.no_grad()
def multi_inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    classes = [3, 2, 3]
    features = ['age', 'gender', 'mask']
    model_dir_list = []

    total_preds_list = []

    for feature, num_class in zip(features, classes):
        print(f"-------{feature}-------")
        num_classes = num_class
        fold_preds_list = []

        model_dir_list = sorted(glob.glob(model_dir + '_' + feature + '*'))
        print(model_dir_list)
        for d in model_dir_list:
            print(args.model_name, args.pth_name, d)
            model = cv_load_model(args.model_name, args.pth_name, d, num_classes, device).to(device)
            if model is None:
                print("model is None")
                continue

            model.eval()

            img_root = os.path.join(data_dir, 'faces')
            info_path = os.path.join(data_dir, 'info.csv')
            info = pd.read_csv(info_path)

            img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
            test_dataset = TestDataset(img_paths, args.resize)

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=8,
                shuffle=False,
                pin_memory=use_cuda,
                drop_last=False,
            )

            print("Calculating inference results..")
            preds = []
            with torch.no_grad():
                pbar = tqdm(test_loader)
                for idx, images in enumerate(pbar):
                    images = images.to(device)
                    pred = model(images)
                    # pred = pred.argmax(dim=-1)
                    preds.extend(pred.cpu().numpy())

            fold_preds_list.append(np.array(preds))

        fold_preds = np.zeros_like(fold_preds_list[0])
        for preds in fold_preds_list:
            fold_preds += preds
        fold_preds = np.argmax(fold_preds, -1)

        if feature == 'gender':
            fold_preds *= 3
        elif feature == 'mask':
            fold_preds *= 6

        total_preds_list.append(np.array(fold_preds))

    total_preds = np.zeros_like(total_preds_list[0])
    for preds in total_preds_list:
        total_preds += preds
    info['ans'] = total_preds
    info.to_csv(os.path.join(output_dir, f'output_{args.pth_name}.csv'), index=False)
    print(f'Inference Done!')


# inference loaded cv train models
@torch.no_grad()
def cv_inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18

    fold_preds_list = []
    model_dir_list = sorted(glob.glob(model_dir + '*'))
    for model_dir in model_dir_list:
        model = cv_load_model(args.model_name, args.pth_name, model_dir, num_classes, device)
        if model is None:
            continue
        model = model.to(device)
        model.eval()

        img_root = os.path.join(data_dir, 'faces')
        info_path = os.path.join(data_dir, 'info.csv')
        info = pd.read_csv(info_path)

        img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
        test_dataset = TestDataset(img_paths, args.resize)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        print("Calculating inference results..")
        preds = []
        with torch.no_grad():
            pbar = tqdm(test_loader)
            for idx, images in enumerate(pbar):
                images = images.to(device)
                pred = model(images)
                # pred = pred.argmax(dim=-1)
                preds.extend(pred.cpu().numpy())

        fold_preds_list.append(np.array(preds))

    fold_preds = np.zeros_like(fold_preds_list[0])
    for preds in fold_preds_list:
        fold_preds += preds
    fold_preds = np.argmax(fold_preds, -1)
    info['ans'] = fold_preds
    info.to_csv(os.path.join(output_dir, f'output_{args.output_name}.csv'), index=False)
    print(f'Inference Done!')


# inference loaded ensemble models
@torch.no_grad()
def ensemble_inference(data_dir, ensemble_model_dir, ensemble_model_name, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18

    fold_preds_list = []
    for model_dir, model_name in zip(ensemble_model_dir, ensemble_model_name):
        model_dir = os.path.join('./model', model_dir)
        model_dir_list = sorted(glob.glob(model_dir + '*'))
        for model_dir in model_dir_list:
            model = cv_load_model(model_name, args.pth_name, model_dir, num_classes, device)
            if model is None:
                continue
            model = model.to(device)
            model.eval()

            img_root = os.path.join(data_dir, 'faces')
            info_path = os.path.join(data_dir, 'info.csv')
            info = pd.read_csv(info_path)

            img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
            test_dataset = TestDataset(img_paths, args.resize)

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=4,
                shuffle=False,
                pin_memory=use_cuda,
                drop_last=False,
            )

            print("Calculating inference results..")
            preds = []
            with torch.no_grad():
                pbar = tqdm(test_loader)
                for idx, images in enumerate(pbar):
                    images = images.to(device)
                    pred = model(images)
                    # pred = pred.argmax(dim=-1)
                    preds.extend(pred.cpu().numpy())

            fold_preds_list.append(np.array(preds))

    fold_preds = np.zeros_like(fold_preds_list[0])
    for preds in fold_preds_list:
        fold_preds += preds
    fold_preds = np.argmax(fold_preds, -1)
    info['ans'] = fold_preds
    info.to_csv(os.path.join(output_dir, f'output_{args.output_name}.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=120, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(280, 210),
                        help='resize size for image when you trained (default: (280, 210))')
    parser.add_argument('--model', type=str, default='Model', help='model type (default: BaseModel)')
    parser.add_argument('--model_name', type=str, default='efficientnet_b4',
                        help='what kinds of models (default: efficientnet_b4)')
    parser.add_argument('--pth_name', type=str, default='', help='which pth you will use (not optional)')
    parser.add_argument('--output_name', type=str, default='', help='output name (not optional)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', ''))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--cv', type=bool, default=False, help='cross validation (default: False)')

    parser.add_argument('--multi_label', type=bool, default=False, help='multi label train (default: False)')

    parser.add_argument('--ensemble', type=bool, default=False, help='ensemble (default: False)')
    parser.add_argument('--ensemble_model_dir', nargs='+', type=str, default=[], help='model_dir for ensemble')
    parser.add_argument('--ensemble_model_name', nargs='+', type=str, default=[], help='model name for ensemble')


    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = os.path.join('./model', args.model_dir)
    output_dir = args.output_dir
    ensemble_model_dir = args.ensemble_model_dir
    ensemble_model_name = args.ensemble_model_name

    os.makedirs(output_dir, exist_ok=True)


    assert args.pth_name, "적용하고자 하는 모델 파라미터를 입력해주세요. cross_validation & multi label & ensemble 시에는 best 로만 입력해 주세요"
    assert args.output_name, "output 이름을 입력해주세요"
    if args.cv:
        assert args.model_dir, "기본경로로 ./model 이 설정되어 있습니다. 하위 경로를 추가로 입력해주세요. cross_validation & multi label 시에는 train 시 name 과 동일"
        cv_inference(data_dir, model_dir, output_dir, args)
    elif args.multi:
        multi_inference(data_dir, model_dir, output_dir, args)
    elif args.ensemble:
        assert args.ensemble_model_dir, "앙상블에 사용할 ./model 하위 폴더명을 공백으로 구분해서 넣어주세요"
        assert args.ensemble_model_name, "앙상블에 사용할 timm 모델명을 공백으로 구분해서 넣어주세요"
        ensemble_inference(data_dir, ensemble_model_dir, ensemble_model_name, output_dir, args)
    else:
        assert args.model_dir, "기본경로로 ./model 이 설정되어 있습니다. 하위 경로를 추가로 입력해주세요. cross_validation & multi label 시에는 train 시 name 과 동일"
        inference(data_dir, model_dir, output_dir, args)
