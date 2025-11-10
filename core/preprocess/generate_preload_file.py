# -*- encoding: utf-8 -*-
"""
@File    :   generate_preload_file.py.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/12/26 15:35   Gzhlaker      1.0         None
"""
import os
import pickle
import numpy as np
import cv2
import glob
import torch
from tqdm import tqdm


def load(p):
    return np.load(p, allow_pickle=True)


def load_info(prefix):
    train_info = load("./phoenix2014/train_info.npy").item()
    test_info = load("./phoenix2014/test_info.npy").item()
    dev_info = load("./phoenix2014/dev_info.npy").item()
    train_info["prefix"] = prefix
    test_info["prefix"] = prefix
    dev_info["prefix"] = prefix
    return train_info, test_info, dev_info


def generate(gloss_dict, info_list):
    for key in tqdm(info_list):
        if key != "prefix":
            source_path = os.path.join(info_list["prefix"], info_list[key]["folder"])
            target_path = source_path.replace("fullFrame-256x256px", "fullFrame-256x256px-pickle")
            print(source_path)
            print(target_path)
            image_list = sorted(glob.glob(source_path))
            image_list = image_list[int(torch.randint(0, 1, [1]))::1]

            label_list = []
            for phase in info_list[key]['label'].split(" "):
                if phase == '':
                    continue
                if phase in gloss_dict.keys():
                    label_list.append(gloss_dict[phase][0])

            image_data = torch.cat(
                [torch.tensor(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)) for img_path in image_list])

            if not os.path.exists(target_path[:-5]):
                os.makedirs(target_path[:-5])
            with open(os.path.join(target_path[:-5], "item.pickle"), mode="wb") as f:
                pickle.dump(
                    [image_data, label_list, info_list[key]],
                    f
                )


def main():
    gloss_dict = np.load("./phoenix2014/gloss_dict.npy", allow_pickle=True).item()
    train_info_list, test_info_list, dev_info_list = load_info(
        "/sda/home/guozihang/guozihang/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px")
    generate(gloss_dict, train_info_list)
    generate(gloss_dict, test_info_list)
    generate(gloss_dict, dev_info_list)


if __name__ == "__main__":
    main()
