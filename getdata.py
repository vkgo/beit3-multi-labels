import os
import shutil
import concurrent.futures

oridatadir = ['/workspace/share/kitreg/data/scene_classify/kitchen', '/workspace/share/clip_head/data/scene_classify/water']
targetdatadir = '/workspace/share/beit3-multi-labels/data'

# 读取oridatadir中各目录中的所有文件夹
# 将文件夹中的所有文件复制到targetdatadir中
# 取oridatadir中目录的最后一级作为标签，例如：/workspace/share/clip_head/data/scene_classify/kitchen的标签为kitchen
# 在/workspace/share/clip_head/data/scene_classify/kitchen中有许多个子文件夹
# 每个子文件夹中都有train、eval、test
# 每个train、eval、test中都有yes和no两个文件夹，yes文件夹中存放的是正样本，no文件夹中存放的是负样本
# 将yes文件夹中的所有文件复制到targetdatadir中，存放的文件夹为kitchen
# 若是train则存放的文件夹为train，eval则存放的文件夹为eval，test则存放的文件夹为test
# 将no文件夹中的所有文件复制到targetdatadir中，存放的文件夹为others
# 若是train则存放的文件夹为train，eval则存放的文件夹为eval，test则存放的文件夹为test
# 其他目录同理

def copy_files(source_path, target_path):
    shutil.copy2(source_path, target_path)

def process_directory(directory):
    label = os.path.basename(directory)
    for subdir in os.listdir(directory):
        if subdir.startswith('.'):
            continue  # skip hidden directories
        for split in ['train', 'eval', 'test']:
            split_dir = os.path.join(directory, subdir, split)
            if not os.path.isdir(split_dir):
                continue  # skip non-existent split directories
            yes_dir = os.path.join(split_dir, 'yes')
            no_dir = os.path.join(split_dir, 'no')
            if os.path.isdir(yes_dir):
                target_yes_dir = os.path.join(targetdatadir, split, label)
                os.makedirs(target_yes_dir, exist_ok=True)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for filename in os.listdir(yes_dir):
                        source_path = os.path.join(yes_dir, filename)
                        target_path = os.path.join(target_yes_dir, filename)
                        executor.submit(copy_files, source_path, target_path)
            if os.path.isdir(no_dir):
                target_no_dir = os.path.join(targetdatadir, split, 'others')
                os.makedirs(target_no_dir, exist_ok=True)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for filename in os.listdir(no_dir):
                        source_path = os.path.join(no_dir, filename)
                        target_path = os.path.join(target_no_dir, filename)
                        executor.submit(copy_files, source_path, target_path)

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for directory in oridatadir:
            executor.submit(process_directory, directory)
