import os
import multiprocessing
from PIL import Image

def worker(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
    except (IOError, SyntaxError) as e:
        print(f"{filepath} cannot be opened: {e}")
        os.remove(filepath)
    try:
        with Image.open(filepath) as img:
            img.load()
    except OSError as e:
        print(f"{filepath} is truncated: {e}")
        os.remove(filepath)

def check_images(directory):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isdir(filepath):
            check_images(filepath)
        else:
            pool.apply_async(worker, args=(filepath,))

    pool.close()
    pool.join()

def check_truncated_images(directory):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isdir(filepath):
            check_truncated_images(filepath)
        else:
            pool.apply_async(worker, args=(filepath,))

    pool.close()
    pool.join()

targetdir = "./data-final-online/online"
check_images(targetdir)
check_truncated_images(targetdir)
