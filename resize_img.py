# +
import os
import cv2
import glob
from pathlib import Path
import multiprocessing
from multiprocessing.dummy import Pool
import concurrent.futures
import glob

import numpy as np
import numpy
from PIL import Image, ImageDraw
import logging
import time

def remove_black_edge_v1(cls, ori_image, color_space_bgr=True, fast_scale=1):
    """From Wangxin's version, <http://172.16.0.2/iKang/model-macular/blob/dev/utils/utils.py>
    """

    ori_image_backup = ori_image

    if fast_scale != 1:
        ori_image = cv2.resize(ori_image, (0,0), fx=1.0/fast_scale, fy=1.0/fast_scale)

    if color_space_bgr:
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(ori_image)

    img_width = ori_image.shape[1]
    img_height = ori_image.shape[0]

    mean_pixel = cls.ave_edge_cal(ori_image)

    if mean_pixel < 127:
        img_mask = np.array(((b < 30) * (g < 30) * (r < 30)) * 1, dtype=np.uint8)
    else:
        img_mask = np.array((((b > 245) * (g > 245) * (r > 245))) * 1, dtype=np.uint8)

    # Cauculate the x-beginning for each line by sum the half mask
    half_start = np.sum(img_mask[:, :img_width // 2], 1).reshape(img_height, 1)
    # Cauculate the x-end by img_width - sum of the right half mask
    half_end = img_width - np.sum(img_mask[:, img_width // 2:], 1).reshape(img_height, 1)

    start_len = img_width // 2 - np.min(half_start)
    end_len = np.max(half_end) - img_width // 2
    half_len = max(start_len, end_len)

    cor_sum = np.sum(img_mask, 1).reshape(img_height, 1)
    cor_sum_binary = np.array((cor_sum > img_width - 10) * 1, dtype=np.uint8)

    vertical_start = cor_sum_binary[:len(cor_sum_binary) // 2]
    vertical_end = cor_sum_binary[len(cor_sum_binary) // 2:]

    vertical_start_cor = np.sum(vertical_start)
    vertical_end_cor = np.sum(vertical_end)

    xmin, ymin, xmax, ymax = cls.regularize_cor_cut(ori_image,
                                                int(img_width // 2 - half_len),
                                                int(vertical_start_cor),
                                                int(img_width // 2 + half_len),
                                                int(img_height - vertical_end_cor))

    if fast_scale != 1:
        xmin, ymin, xmax, ymax = int(fast_scale*xmin), int(fast_scale*ymin), \
                                 int(fast_scale*xmax), int(fast_scale*ymax)

    ori_image = ori_image_backup
    cut_image = ori_image[ymin: ymax, xmin: xmax]

    return cut_image, ymin, ymax, xmin, xmax
def ave_edge_cal(ori_image):
    gray = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
    ul = gray[:5, :5]
    ur = gray[:5, ori_image.shape[1] - 5:]
    ll = gray[ori_image.shape[0] - 5:, :5]
    lr = gray[ori_image.shape[0] - 5:, ori_image.shape[1] - 5:]
    mean = (np.sum(ul) / 25 + np.sum(ur) / 25 + np.sum(ll) / 25 + np.sum(lr) / 25) / 4
    return mean
def regularize_cor_cut(ori_image, xmin, ymin, xmax, ymax):
    img_height = ori_image.shape[0]
    img_width = ori_image.shape[1]
    # Mask sure all coordinates in right area

    if xmin > img_width:
        xmin = img_width
    elif xmin < 0:
        xmin = 0

    if xmax > img_width:
        xmax = img_width
    elif xmax < 0:
        xmax = 0

    if ymin > img_height:
        ymin = img_height
    elif ymin < 0:
        ymin = 0

    if ymax > img_height:
        ymax = img_height
    elif ymax < 0:
        ymax = 0

    return xmin, ymin, xmax, ymax
def create_circular_mask(h, w, center=None, radius=None, reverse=False):
    """
    adapted from <https://stackoverflow.com/questions/44865023/circular-masking-an-image-in-python-using-numpy-arrays>
    mask 1 in circle
    :param h:
    :param w:
    :param center:
    :param radius:
    :return:
    """


    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    if not reverse:
        mask = dist_from_center <= radius
    else:
        mask = dist_from_center > radius

    return mask
def _remove_black_edge(path):
    ori_image = cv2.imread(path)
#     ori_image = np.float32(ori_image)
    b, g, r = cv2.split(ori_image)
    img_width = ori_image.shape[1]
    img_height = ori_image.shape[0]

    mean_pixel = ave_edge_cal(ori_image)

    if mean_pixel < 127:
        img_mask = np.array(((b < 30) * (g < 30) * (r < 30)) * 1, dtype=np.uint8)
    else:
        img_mask = np.array((((b > 245) * (g > 245) * (r > 245))) * 1, dtype=np.uint8)

    # Cauculate the x-beginning for each line by sum the half mask
    half_start = np.sum(img_mask[:, :int(img_width/2)], 1).reshape(img_height, 1)
    # Cauculate the x-end by img_width - sum of the right half mask
    half_end = img_width - np.sum(img_mask[:, int(img_width/2):], 1).reshape(img_height, 1)

    start_len = img_width // 2  - np.min(half_start)
    end_len = np.max(half_end) - img_width // 2 
    half_len = max(start_len,end_len)

    cor_sum = np.sum(img_mask, 1).reshape(img_height, 1)
    cor_sum_binary = np.array((cor_sum > img_width - 10)*1, dtype=np.uint8)

    vertical_start = cor_sum_binary[:len(cor_sum_binary)// 2]
    vertical_end = cor_sum_binary[len(cor_sum_binary) // 2:]

    vertical_start_cor = np.sum(vertical_start)
    vertical_end_cor = np.sum(vertical_end)

    xmin, ymin, xmax, ymax = regularize_cor_cut(ori_image, int(img_width // 2 - half_len), int(vertical_start_cor), int(img_width // 2 + half_len), int(img_height-vertical_end_cor))
    cut_image = ori_image[ymin: ymax, xmin: xmax]

    return cut_image

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            # if f.endswith('.xls'):
            fullname = os.path.join(root, f)
            yield fullname

# def main():
#     # base = '/mnt/workdir/fengwei/yifuying/pih/img_resize/'
#     base = '/mnt/workdir/fengwei/yifuying/bingshi_data/normal_image/'
#     width = 256
#     height = 256
#     for idx in findAllFile(base):
#         # print(idx)
#         image_name = idx.split("/")[-1]
#         # print(image_name)
#         src = cv2.imread(idx, cv2.IMREAD_COLOR)
#         # src = cv2.imread(jj, cv2.IMREAD_ANYCOLOR)
#         dst = cv2.resize(src, dsize=(width, height), interpolation=cv2.INTER_AREA)
#         # dst = cv2.resize(src, dsize=(width, height), interpolation=cv2.INTER_AREA)
#         # dst=cv2.resize(src, dsize=None,fx=0.5,fy=0.5, interpolation=cv2.INTER_AREA)
#         cv2.imwrite(idx,dst)

# if __name__ == '__main__':
#     main()


# -

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.jpg'):
                fullname = os.path.join(root, f)
                yield fullname

# +
# base = '/mnt/workdir/fengwei/yifuying/pih/img_resize/'
base = '/mnt/workdir/fengwei/xiapeng/cognitive_guanxin/img/2022/01/08/'
width = 512
height = 512
for idx in findAllFile(base):
    print(idx)
    image_name = idx.replace("/img/","/img_resize/")
    file_path, filename = os.path.split(image_name)
    os.makedirs(file_path) if not os.path.exists(file_path) else None
    print(file_path)
    
    try:
        src = _remove_black_edge(idx)
        dst = cv2.resize(src, dsize=(width, height), interpolation=cv2.INTER_AREA)
        # dst = cv2.resize(src, dsize=(width, height), interpolation=cv2.INTER_AREA)
        # dst=cv2.resize(src, dsize=None,fx=0.5,fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_name,dst)
    except:
        src = cv2.imread(idx, cv2.IMREAD_COLOR)
        dst = cv2.resize(src, dsize=(width, height))
        cv2.imwrite(image_name,dst)
        
#     src = cv2.imread(idx, cv2.IMREAD_COLOR)
#     # src = cv2.imread(jj, cv2.IMREAD_ANYCOLOR)
#     dst = cv2.resize(src, dsize=(width, height), interpolation=cv2.INTER_AREA)
#     # dst = cv2.resize(src, dsize=(width, height), interpolation=cv2.INTER_AREA)
#     # dst=cv2.resize(src, dsize=None,fx=0.5,fy=0.5, interpolation=cv2.INTER_AREA)
#     cv2.imwrite(image_name,dst)
