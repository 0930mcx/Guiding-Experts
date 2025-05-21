# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from PIL import Image
from timm.data.auto_augment import augment_and_mix_transform, rand_augment_transform
from timm.data.random_erasing import RandomErasing
from timm.data.transforms_factory import transforms_noaug_train, transforms_imagenet_eval
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup, auto_augment_transform


from utils.MyTransforms import CustomDualTransform


from .cached_image_folder import CachedImageFolder

from .samplers import SubsetRandomSampler

try:
    from torchvision.transforms import InterpolationMode, ToPILImage


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        # shuffle=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        # shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


class ImageNetDataset(Dataset):
    def __init__(self, dataset_dir):
        """
        :param dataset_dir: 数据根目录，每个子文件夹对应一个类别
        :param transform: 图像变换
        """
        self.dataset_dir = dataset_dir
        self.transform = CustomDualTransform(img_size=224, interpolation="bicubic", hflip=0.5, vflip=0, re_mode="pixel", re_prob=0.25, re_count=1)
        self.file_list = []  # 保存所有文件路径
        self.class_to_idx = {}  # 文件夹名称到类别 ID 的映射
        self.class_samples = {}
        self.to_pil = ToPILImage()
        # 获取所有类别文件夹
        class_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
        class_folders.sort()
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_folders)}  # 为每个类别分配一个 ID
        self.transform_for_mask = transforms.Compose([transforms.Resize((192, 192)),
                            transforms.ToTensor(),

                                                      ])
        # 遍历每个类别文件夹，获取 .pt 文件路径
        for class_folder in class_folders:
            class_folder_path = os.path.join(dataset_dir, class_folder)
            class_files = [os.path.join(class_folder_path, f) for f in os.listdir(class_folder_path) if
                           f.endswith('.pth')]
            self.class_samples[class_folder] = class_files
            self.file_list.extend(
                [(file, self.class_to_idx[class_folder]) for file in class_files]
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # print(self.transform)
        file_path, target = self.file_list[idx]
        # print(file_path)
        data = torch.load(file_path, map_location='cpu', weights_only=True)
        # data = self.to_pil(data)
        image, mask = torch.split(data, [3, 1], dim=0)
        # print(f"start {x.shape}")
        if self.transform:
            image, mask = self.transform(self.to_pil(image), self.to_pil(mask))
        return image, mask, target
class ImageNetDataset2(Dataset):
    def __init__(self, dataset_dir, mask_dir, format=".JPEG"):
        """
        :param dataset_dir: 数据根目录，每个子文件夹对应一个类别
        :param mask_dir: 掩码文件所在的目录，包含与 dataset_dir 中图片同名的 .pth 文件
        :param transform: 图像变换
        """
        self.dataset_dir = dataset_dir
        self.mask_dir = mask_dir
        self.transform = CustomDualTransform(img_size=224, interpolation="bicubic", hflip=0.5, vflip=0, re_mode="pixel", re_prob=0.25, re_count=1)
        self.file_list = []  # 保存所有文件路径
        self.class_to_idx = {}  # 文件夹名称到类别 ID 的映射
        self.class_samples = {}
        self.to_pil = ToPILImage()

        # 获取所有类别文件夹
        class_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
        class_folders.sort()
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_folders)}  # 为每个类别分配一个 ID

        # 遍历每个类别文件夹，获取 .JPEG 文件路径及对应的 .pth 文件路径
        for class_folder in class_folders:
            class_folder_path = os.path.join(dataset_dir, class_folder)
            mask_folder_path = os.path.join(mask_dir, class_folder)  # 获取对应类别的掩码文件夹路径
            class_files = [f for f in os.listdir(class_folder_path) if f.endswith(format)]

            self.class_samples[class_folder] = class_files
            for file in class_files:
                image_path = os.path.join(class_folder_path, file)
                mask_path = os.path.join(mask_folder_path, file.replace(format, '.pth'))  # 对应的 mask 路径

                if os.path.exists(mask_path):  # 确保掩码文件存在
                    self.file_list.append((image_path, mask_path, self.class_to_idx[class_folder]))
                else:
                    print(f"Warning: Mask file {mask_path} for image {image_path} does not exist.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_path, mask_path, target = self.file_list[idx]

        # 读取图片
        image = Image.open(image_path).convert("RGB")

        # 读取掩码数据 (.pth 文件)
        mask = torch.load(mask_path, map_location='cpu', weights_only=True)  # 读取 .pth 文件
        # mask = mask_data.squeeze(0)  # 去掉多余的维度，假设 mask 数据是 [1, H, W]

        # 应用图像和掩码变换
        if self.transform:
            # 使用自定义的 transform 函数进行变换
            image, mask = self.transform(image, self.to_pil(mask))

        return image, mask, target
def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            if is_train:
                train_dir = os.path.join(config.DATA.DATA_PATH, 'train')
                mask_dir = os.path.join(config.DATA.DATA_PATH, 'train_mask')
                dataset = ImageNetDataset2(
                    train_dir,
                    mask_dir,
                )
            else:
                root = os.path.join(config.DATA.DATA_PATH, 'val4')
                dataset = datasets.ImageFolder(root, transform=transform)

        # prefix = 'train' if is_train else 'val'
        # dataset_dir = os.path.join(config.DATA.DATA_PATH, prefix)  # dataset_dir 是存储 .pt 文件的文件夹
        #
        # # files = [f for f in os.listdir(dataset_dir) if f.endswith('.pt')]
        # # files.sort()  # 确保按顺序加载
        # # dataset = ImageNetDataset(dataset_dir, transform)
        # dataset = ImageNetDataset(dataset_dir, transform)  # every class 300 samples
        nb_classes = 1000
    elif config.DATA.DATASET == 'stanford_cars':
        if is_train:
            train_dir = os.path.join(config.DATA.DATA_PATH, 'train')
            mask_dir = os.path.join(config.DATA.DATA_PATH, 'train_mask')
            dataset = ImageNetDataset2(
                train_dir,
                mask_dir,
                format=".jpg"
            )
            print(len(dataset))
        else:
            root = os.path.join(config.DATA.DATA_PATH, 'test')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 196
    elif config.DATA.DATASET == 'pets':
        if is_train:
            train_dir = os.path.join(config.DATA.DATA_PATH, 'train')
            mask_dir = os.path.join(config.DATA.DATA_PATH, 'train_mask')
            dataset = ImageNetDataset2(
                train_dir,
                mask_dir,
                format=".jpg"
            )
            print(len(dataset))
        else:
            root = os.path.join(config.DATA.DATA_PATH, 'test')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 37
    elif config.DATA.DATASET == 'painting':
        if is_train:
            train_dir = os.path.join(config.DATA.DATA_PATH, 'train')
            mask_dir = os.path.join(config.DATA.DATA_PATH, 'train_mask')
            dataset = ImageNetDataset2(
                train_dir,
                mask_dir,
                format=".jpg"
            )
            print(len(dataset))
        else:
            root = os.path.join(config.DATA.DATA_PATH, 'test')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 345
    elif config.DATA.DATASET == 'sketch':
        if is_train:
            train_dir = os.path.join(config.DATA.DATA_PATH, 'train')
            mask_dir = os.path.join(config.DATA.DATA_PATH, 'train_mask')
            dataset = ImageNetDataset2(
                train_dir,
                mask_dir,
                format=".jpg"
            )
            print(len(dataset))
        else:
            root = os.path.join(config.DATA.DATA_PATH, 'test')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 345
    elif config.DATA.DATASET == 'clipart':
        if is_train:
            train_dir = os.path.join(config.DATA.DATA_PATH, 'train')
            mask_dir = os.path.join(config.DATA.DATA_PATH, 'train_mask')
            dataset = ImageNetDataset2(
                train_dir,
                mask_dir,
                format=".jpg"
            )
            print(len(dataset))
        else:
            root = os.path.join(config.DATA.DATA_PATH, 'test')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 345
    elif config.DATA.DATASET == 'mini-imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            # dataset = datasets.ImageFolder(root, transform=transform)

            if is_train:
                dataset = ImageNetDataset(root)
                # dataset = load_ImageNet3("/ai_home/data/private/mcx/dataset/mini-imagenet-sam")
            else:
                dataset = datasets.ImageFolder("/ai_home/data/private/mcx/dataset/Mini-ImageNet-Dataset/val", transform=transform)
                # masks_data = torch.load("/ai_home/data/private/mcx/dataset/new_sam_mini/val2.pth")
            #     print(f"mask train data: {len(masks_data)}")
            # # else:
            # #     masks_data = torch.load("/ai_home/data/private/mcx/dataset/new_sam_mini/val2.pth")
            #     # masks_data = torch.load("/ai_home/data/private/mcx/dataset/new_sam_mini/val_front_masks.pth")
            # #     # print(f"mask val data: {len(masks_data)}")
            # #     dataset = ModifiedDataset(dataset, masks_data)
            #     dataset = ModifiedDataset(dataset, masks_data)
            #     logger.info(f"succeed load masks and dataset: {len(dataset)}")
        nb_classes = 100
    elif config.DATA.DATASET == 'imagenet-100':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            if is_train:
                train_dir = os.path.join(config.DATA.DATA_PATH, 'train')
                mask_dir = os.path.join(config.DATA.DATA_PATH, 'train_mask')
                dataset = ImageNetDataset2(
                    train_dir,
                    mask_dir,
                )
            else:
                root = os.path.join(config.DATA.DATA_PATH, 'val')
                dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 100
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes

# def build_dataset(is_train, config):
#     transform = build_transform(is_train, config)
#     if config.DATA.DATASET == 'imagenet':
#         prefix = 'train' if is_train else 'val'
#         print(f"进入imagenet")
#         if config.DATA.ZIP_MODE:
#             ann_file = prefix + "_map.txt"
#             prefix = prefix + ".zip@/"
#             dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
#                                         cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
#         else:
#             root = os.path.join(config.DATA.DATA_PATH, prefix)
#             dataset = datasets.ImageFolder(root, transform=transform)
#             # if is_train:
#             #     masks_data = torch.load("/ai_home/data/private/mcx/dataset/imagenet_sam_mini/train_192.pth")
#             #     print(f"mask train data: {len(masks_data)}")
#             #     # masks_data = F.interpolate(masks_data.float(), size=(224, 224), mode='nearest').byte()
#             # # else:
#             # #     # masks_data = torch.load("/ai_home/data/private/mcx/swin-transformer/sam_imagenet/all_tensor_val.pth")
#             # #     masks_data = torch.load("/ai_home/data/private/mcx/dataset/new_sam_mini/val2.pth")
#             # #     print(f"mask val data: {len(masks_data)}")
#             #     dataset = ModifiedDataset(dataset, masks_data)
#             # logger.info(f"succeed load masks and dataset: {len(dataset)}")
#         nb_classes = 1000
#     elif config.DATA.DATASET == 'mini-imagenet':
#         prefix = 'train' if is_train else 'val'
#         if config.DATA.ZIP_MODE:
#             ann_file = prefix + "_map.txt"
#             prefix = prefix + ".zip@/"
#             dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
#                                         cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
#         else:
#             root = os.path.join(config.DATA.DATA_PATH, prefix)
#             dataset = datasets.ImageFolder(root, transform=transform)
#             if is_train:
#                 masks_data = torch.load("/ai_home/data/private/mcx/dataset/new_sam_mini/train_front_masks.pth")
#                 print(f"mask train data: {len(masks_data)}")
#             # else:
#             #     masks_data = torch.load("/ai_home/data/private/mcx/dataset/new_sam_mini/val2.pth")
#                 # masks_data = torch.load("/ai_home/data/private/mcx/dataset/new_sam_mini/val_front_masks.pth")
#             #     # print(f"mask val data: {len(masks_data)}")
#             #     dataset = ModifiedDataset(dataset, masks_data)
#                 dataset = ModifiedDataset(dataset, masks_data)
#                 logger.info(f"succeed load masks and dataset: {len(dataset)}")
#         nb_classes = 100
#     elif config.DATA.DATASET == 'imagenet22K':
#         prefix = 'ILSVRC2011fall_whole'
#         if is_train:
#             ann_file = prefix + "_map_train.txt"
#         else:
#             ann_file = prefix + "_map_val.txt"
#         dataset = IN22KDATASET(config.DATA.DATA_PATH, ann_file, transform)
#         nb_classes = 21841
#     else:
#         raise NotImplementedError("We only support ImageNet Now.")
#
#     return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform2(
            input_size=192,
            is_training=True,
            color_jitter=0.4,
            # auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25,
            re_mode='const',
            re_count=1,
            interpolation='bicubic',
        )
        # print(transform)
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
def transforms_imagenet_train2(
        img_size=224,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='random',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        separate=False,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3./4., 4./3.))  # default imagenet ratio range
    primary_tfl = []
    if hflip > 0.:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]
    # primary_tfl = []
    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = timm_transforms.str_to_pil_interp(interpolation)
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]
    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [timm_transforms.ToNumpy()]
    else:
        final_tfl += [
            transforms.ToTensor(),

        ]
        if re_prob > 0.:
            final_tfl.append(
                RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu'))

    if separate:
        return transforms.Compose(primary_tfl), transforms.Compose(secondary_tfl), transforms.Compose(final_tfl)
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)
def create_transform2(
        input_size,
        is_training=False,
        use_prefetcher=False,
        no_aug=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        crop_pct=None,
        tf_preprocessing=False,
        separate=False):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if tf_preprocessing and use_prefetcher:
        assert not separate, "Separate transforms not supported for TF preprocessing"
        from timm.data.tf_preprocessing import TfPreprocessTransform
        transform = TfPreprocessTransform(
            is_training=is_training, size=img_size, interpolation=interpolation)
    else:
        if is_training and no_aug:
            assert not separate, "Cannot perform split augmentation with no_aug"
            transform = transforms_noaug_train(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std)
        elif is_training:
            transform = transforms_imagenet_train2(
                img_size,
                scale=scale,
                ratio=ratio,
                hflip=hflip,
                vflip=vflip,
                color_jitter=color_jitter,
                auto_augment=auto_augment,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
                re_prob=re_prob,
                re_mode=re_mode,
                re_count=re_count,
                re_num_splits=re_num_splits,
                separate=separate)
        else:
            assert not separate, "Separate transforms not supported for validation preprocessing"
            transform = transforms_imagenet_eval(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
                crop_pct=crop_pct)

    return transform