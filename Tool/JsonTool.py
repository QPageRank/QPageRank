import os
import json
import chardet
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        # 保留 None 值，不抛出错误
        return super(NumpyEncoder, self).default(obj)


def read(filepath):
    encoding = detect(filepath)
    with open(filepath, "r", encoding=encoding) as file:
        data = json.load(file)
    return data


def modify(filepath, node_num, block_num):
    data = None
    encoding = detect(filepath)
    with open(filepath, "r", encoding=encoding) as file:
        data = json.load(file)
        data["roadselect"]["global"]["node_num"] = node_num
        data["roadselect"]["global"]["block_num"] = block_num
    with open(filepath, "w", encoding=encoding) as file:
        json.dump(data, file, indent=4)


def save(filepath, data):
    # 获取文件所在的目录
    dir_path = os.path.dirname(filepath)

    # 如果目录不存在，则创建
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 保存数据
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_match(save_path, data):
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(data, file, cls=NumpyEncoder, indent=4)


def saveresult(filepath, key, value):
    data = None
    try:
        encoding = detect(filepath)
    except:
        encoding = "UTF-8"
    try:
        with open(filepath, "r", encoding=encoding) as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}  # 如果文件不存在，初始化为空字典
    except json.JSONDecodeError:
        print(f"Error: The file {filepath} is not a valid JSON file.")
        return

    data[key] = value

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding=encoding) as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def detect(file_path):
    with open(file_path, "rb") as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"]
    return encoding
