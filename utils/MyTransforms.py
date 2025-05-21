import torch
from timm.data import str_to_pil_interp

from torchvision.transforms import transforms

from RandomAugment import AugmentOp, rand_augment_transform
from RandomErasing import RandomErasing
from RandomFlip import RandomHorizontalFlip, RandomVerticalFlip
from RandomResizedCropAndInterolation import RandomResizedCropAndInterpolation
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# 示例：创建一个对两个输入图像同时处理的transform
class CustomDualTransform:
    def __init__(self, img_size, interpolation, hflip=0, vflip=0, re_prob=0, re_mode='const', re_count=1, re_num_splits=0, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
        scale = tuple((0.08, 1.0))  # default imagenet scale range
        ratio = tuple((3. / 4., 4. / 3.))  # default imagenet ratio range
        aa_params = dict(
            translate_const=int(img_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = str_to_pil_interp(interpolation)
        self.hflip = hflip
        self.vflip = vflip
        self.resize_transform = RandomResizedCropAndInterpolation(img_size, scale=scale, ratio=ratio, interpolation=interpolation)
        self.hflip_transform = RandomHorizontalFlip(p=hflip)
        self.vflip_transform = RandomVerticalFlip(p=vflip)
        self.random_augment = rand_augment_transform('rand-m9-mstd0.5-inc1', aa_params)
        self.tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(
                                            mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                                            std=torch.tensor(IMAGENET_DEFAULT_STD))
        self.erase_transform = RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu')
    def __call__(self, images, masks):
        """
        对两个输入同时进行处理，例如：
        - input1：图像
        - input2：掩码/标签
        """
        # 假设这里我们做的是Resize和ToTensor
        images, masks = self.resize_transform(images, masks)
        if self.hflip>0:
            images, masks = self.hflip_transform(images, masks)
        if self.vflip>0:
            images, masks = self.vflip_transform(images, masks)
        # print(self.random_augment)
        images, masks  = self.random_augment(images, masks)

        images = self.tensor_transform(images)
        masks = self.tensor_transform(masks)
        images = self.normalize_transform(images)
        images, masks = self.erase_transform(images, masks)
        return images, masks

