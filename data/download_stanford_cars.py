# import kagglehub
#
# # Download latest version
# path = kagglehub.dataset_download("mei1963/domainnet")
#
# print("Path to dataset files:", path)
import os
import shutil
from tqdm import tqdm

def organize_by_split(source_dir, split_txt, output_root):
    with open(split_txt, 'r') as f:
        lines = f.readlines()

    split_name = os.path.splitext(os.path.basename(split_txt))[0]  # train or test
    output_dir = os.path.join(output_root, split_name)
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for line in tqdm(lines, desc=f'Copying {split_name} set'):
        img_rel_path, label = line.strip().split()
        class_name = os.path.dirname(img_rel_path)  # e.g., "airplane"
        src_path = os.path.join(source_dir, img_rel_path)
        dst_dir = os.path.join(output_dir, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src_path, os.path.join(dst_dir, os.path.basename(img_rel_path)))
        count += 1

    print(f"✅ 完成 {split_name}：共 {count} 张图像复制到 {output_dir}")

# === 示例使用 ===
if __name__ == '__main__':
    source_dir = '/data3/ai_home/jiaxinlei/data/private/mcx/dataset/DomainNet'         # 原始图像目录
    train_txt = '/data3/ai_home/jiaxinlei/data/private/mcx/dataset/DomainNet/sketch_train.txt'          # 训练划分文件
    test_txt = '/data3/ai_home/jiaxinlei/data/private/mcx/dataset/DomainNet/sketch_test.txt'            # 测试划分文件
    output_dir = '/data3/ai_home/jiaxinlei/data/private/mcx/dataset/DomainNet2/sketch'                    # 输出根目录

    organize_by_split(source_dir, train_txt, output_dir)
    organize_by_split(source_dir, test_txt, output_dir)


# import os
# import shutil
# from PIL import Image
# import torch
# from tqdm import tqdm
# import numpy as np
# def load_image_names(txt_path):
#     """
#     只提取图像名（不含扩展名）
#     """
#     with open(txt_path, 'r') as f:
#         lines = f.readlines()
#     image_names = [line.strip().split()[0] for line in lines if line.strip() and not line.startswith('#')]
#     return image_names
#
# def convert_mask_to_binary(mask_path):
#     """
#     将掩码中值为1/2转换为1（前景），3转换为0（背景）
#     """
#     mask = Image.open(mask_path)  # 强制转灰度，避免调色板
#     mask_np = np.array(mask, dtype=np.uint8)
#     # print(mask_np)
#     # 显式判断：1或2为前景，3为背景
#     binary_mask = ((mask_np == 1) | (mask_np == 3)).astype(np.uint8)
#     return torch.from_numpy(binary_mask)
#
# def organize_pets_dataset(base_dir):
#     image_dir = os.path.join(base_dir, 'images')
#     ann_dir = os.path.join(base_dir, 'annotations')
#     trimap_dir = os.path.join(ann_dir, 'trimaps')
#
#     train_list = load_image_names(os.path.join(ann_dir, 'trainval.txt'))
#     test_list = load_image_names(os.path.join(ann_dir, 'test.txt'))
#
#     train_out = os.path.join(base_dir, 'train')
#     test_out = os.path.join(base_dir, 'test')
#     mask_out = os.path.join(base_dir, 'train_mask')
#
#     os.makedirs(train_out, exist_ok=True)
#     os.makedirs(test_out, exist_ok=True)
#     os.makedirs(mask_out, exist_ok=True)
#
#     print("📁 正在处理训练集和掩码...")
#     for img_name in tqdm(train_list):
#         img_file = img_name + '.jpg'
#         mask_file = img_name + '.png'
#
#         # 获取类别名
#         class_name = '_'.join(img_name.split('_')[:-1])
#         class_dir = os.path.join(train_out, class_name)
#         os.makedirs(class_dir, exist_ok=True)
#
#         # 拷贝图像
#         shutil.copyfile(os.path.join(image_dir, img_file), os.path.join(class_dir, img_file))
#         # 掩码保存路径
#         mask_class_dir = os.path.join(mask_out, class_name)
#         os.makedirs(mask_class_dir, exist_ok=True)
#         dst_mask_path = os.path.join(mask_class_dir, img_name + '.pth')
#
#         # 转换保存
#         mask_tensor = convert_mask_to_binary(os.path.join(trimap_dir, mask_file))
#         torch.save(mask_tensor, dst_mask_path)
#         # return
#
#     # print("📁 正在处理测试集...")
#     # for img_name in tqdm(test_list):
#     #     img_file = img_name + '.jpg'
#     #     class_name = '_'.join(img_name.split('_')[:-1])
#     #     class_dir = os.path.join(test_out, class_name)
#     #     os.makedirs(class_dir, exist_ok=True)
#     #     shutil.copyfile(os.path.join(image_dir, img_file), os.path.join(class_dir, img_file))
#
#     print("✅ 数据整理完毕！")
#
# # 示例运行
# if __name__ == '__main__':
#     base = '/data3/ai_home/jiaxinlei/data/private/mcx/dataset/pets'  # 请替换为你的实际路径
#     organize_pets_dataset(base)


