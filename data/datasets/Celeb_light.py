import os
import re
import glob
import h5py
import random
import math
import logging
import numpy as np
import os.path as osp
from scipy.io import loadmat

from tools.utils import mkdir_if_missing, write_json, read_json


class Celeb_light(object):
    """ Celeb-reID-light

    Reference:
        Huang et al. Celebrities-ReID: A Benchmark for Clothes Variation in Long-Term Person Re-Identification. IJCNN, 2019.

    URL: https://naiq.github.io/LTCC_Perosn_ReID.html#
    """
    dataset_dir = 'Celeb-reID-light'

    def __init__(self, root='data', meta_dir='PAR_PETA_105.txt', meta_dims=105, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.meta_dir = meta_dir
        self.meta_dims = meta_dims
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self._check_before_run()

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir)
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            self._process_dir_test(self.query_dir, self.gallery_dir)
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_test_imgs = num_query_imgs + num_gallery_imgs
        num_total_clothes = num_train_clothes + num_test_clothes

        print("=> Celeb loaded")
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # clothes")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        print("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
        print("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        print("  ----------------------------------------")
        print("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        print("  ----------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.num_test_clothes = num_test_clothes
        self.num_query_imgs = num_query_imgs
        self.pid2clothes = pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir_train(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_(\d+)')
        pattern2 = re.compile(r'(\w+)_')

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)

        imgdir2attribute = {}
        with open(os.path.join(self.dataset_dir, self.meta_dir), 'r') as f:
            for line in f:
                imgdir, attribute_id, is_present = line.split()

                img_name = osp.basename(imgdir)
                if img_name not in imgdir2attribute:
                    imgdir2attribute[img_name] = [0 for i in range(self.meta_dims)]
                imgdir2attribute[img_name][int(attribute_id)] = int(is_present)


        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id: label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        dataset = []
        images_info = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_path in img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes = pattern2.search(img_path).group(1)

            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]


            img_name = osp.basename(img_path)

            attrs = imgdir2attribute.get(img_name, [0] * self.meta_dims)

            dataset.append((img_path, pid, camid, clothes_id, attrs))
            images_info.append({'attributes': attrs})


            pid2clothes[pid, clothes_id] = 1

        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes

    def _process_dir_test(self, query_path, gallery_path):
        query_img_paths = glob.glob(osp.join(query_path, '*.jpg'))
        gallery_img_paths = glob.glob(osp.join(gallery_path, '*.jpg'))
        query_img_paths.sort()
        gallery_img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_(\d+)')
        pattern2 = re.compile(r'(\w+)_')

        pid_container = set()
        clothes_container = set()

        for img_path in query_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        for img_path in gallery_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)

        imgdir2attribute = {}
        with open(os.path.join(self.dataset_dir, self.meta_dir), 'r') as f:
            for line in f:
                imgdir, attribute_id, is_present = line.split()

                img_name = osp.basename(imgdir)
                if img_name not in imgdir2attribute:
                    imgdir2attribute[img_name] = [0 for i in range(self.meta_dims)]
                imgdir2attribute[img_name][int(attribute_id)] = int(is_present)


        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id: label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        query_dataset = []
        gallery_dataset = []
        images_info_query = []
        images_info_gallery = []

        for img_path in query_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)

            clothes_id = clothes2label[clothes_id]


            img_name = osp.basename(img_path)
            attrs = imgdir2attribute.get(img_name, [0] * self.meta_dims)
            query_dataset.append((img_path, pid, 1, clothes_id, attrs))
            images_info_query.append({'attributes': attrs})


        for img_path in gallery_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            # camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]


            img_name = osp.basename(img_path)
            attrs = imgdir2attribute.get(img_name, [0] * self.meta_dims)
            gallery_dataset.append((img_path, pid, 0, clothes_id, attrs))
            images_info_gallery.append({'attributes': attrs})


        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes


if __name__ == '__main__':

    dataset = Celeb_light(root='/home/ykding/dataset/')
    print(f"Train Clothes: {dataset.num_train_clothes}")
    print(f"Test Clothes: {dataset.num_test_clothes}")