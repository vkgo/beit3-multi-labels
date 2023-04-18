# datatdir 目录下有imagenet.train.index.jsonl, imagenet.test.index.jsonl, imagenet.val.index.jsonl
# 分别是训练集，测试集，验证集的标签文件
# 格式为：{"image_path": "test/kitchen/000000001526.jpg", "label": 0}
# 这是一个单分类的数据集，label只有0,1,2
# 需要构造一个多分类的假数据集，label有0,1,2,3,4
# 采取One-Hot编码的方式
# 格式为：{"image_path": "test/kitchen/000000001526.jpg", "label": [0, 1, 0, 0, 0]}
# 假数据集的图片路径不变，只是修改标签
# 生成的假数据集保存在fakedatadir目录下
import json
import os
import random
import shutil

datatdir = './data'
fakedatadir = './fakedata'

# 删除 fakedatadir 目录，如果已存在
if os.path.exists(fakedatadir):
    shutil.rmtree(fakedatadir)

# 创建 fakedatadir 目录
os.makedirs(fakedatadir)

def random_multi_label(num_classes):
    mask = [random.choice([0, 1]) for _ in range(num_classes)]
    label = [random.choice([0, 1]) if mask[i] == 1 else 0 for i in range(num_classes)]

    return label, mask

# 遍历所有数据集
for dataset in ['train', 'test', 'val']:
    # 读取原始标签文件
    with open(os.path.join(datatdir, f'imagenet.{dataset}.index.jsonl'), 'r', encoding='utf-8') as infile:
        # 创建一个新的标签文件，用于保存多分类的假数据集标签
        with open(os.path.join(fakedatadir, f'imagenet.{dataset}.index.jsonl'), 'w', encoding='utf-8') as outfile:
            # 遍历原始标签文件的每一行
            for line in infile:
                # 解析 JSON 对象
                data = json.loads(line)

                # 生成随机多标签和对应的 mask
                multi_label, mask = random_multi_label(5)

                # 更新标签和添加 mask
                data['label'] = multi_label
                data['mask'] = mask

                # 将新的 JSON 对象写入新的标签文件
                outfile.write(json.dumps(data) + '\n')