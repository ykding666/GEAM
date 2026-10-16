import os
import glob
import logging
import numpy as np
import os.path as osp
import random


class LaST(object):
    """ LaST
    完全对齐原始格式要求，并修复了路径匹配导致的 KeyError
    """
    dataset_dir = "last"

    def __init__(self, root='data', meta_dir='PAR_PETA_105.txt', meta_dims=105, **kwargs):
        super(LaST, self).__init__()
        self.root = root
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.meta_dir = meta_dir
        self.meta_dims = meta_dims

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_query_dir = osp.join(self.dataset_dir, 'val', 'query')
        self.val_gallery_dir = osp.join(self.dataset_dir, 'val', 'gallery')
        self.test_query_dir = osp.join(self.dataset_dir, 'test', 'query')
        self.test_gallery_dir = osp.join(self.dataset_dir, 'test', 'gallery')

        self._check_before_run()

        # 1. 预处理
        pid2label, clothes2label, pid2clothes = self.get_pid2label_and_clothes2label(self.train_dir)

        # 2. 处理各目录
        train, num_train_pids = self._process_dir(self.train_dir, pid2label=pid2label, clothes2label=clothes2label,
                                                  relabel=True)
        val_query, num_val_query_pids = self._process_dir(self.val_query_dir, relabel=False)
        val_gallery, num_val_gallery_pids = self._process_dir(self.val_gallery_dir, relabel=False, recam=len(val_query))
        test_query, num_test_query_pids = self._process_dir(self.test_query_dir, relabel=False)
        test_gallery, num_test_gallery_pids = self._process_dir(self.test_gallery_dir, relabel=False,
                                                                recam=len(test_query))

        # 3. 统计信息
        num_total_pids = num_train_pids + num_val_gallery_pids + num_test_gallery_pids
        num_total_imgs = len(train) + len(val_query) + len(val_gallery) + len(test_query) + len(test_gallery)

        logger = logging.getLogger('reid.dataset')
        logger.info("=> LaST loaded")
        logger.info("Dataset statistics:")
        logger.info("  --------------------------------------------")
        logger.info("  subset        | # ids | # images | # clothes")
        logger.info("  ----------------------------------------")
        logger.info("  train         | {:5d} | {:8d} | {:9d}".format(num_train_pids, len(train), len(clothes2label)))
        logger.info("  query(val)    | {:5d} | {:8d} |".format(num_val_query_pids, len(val_query)))
        logger.info("  gallery(val)  | {:5d} | {:8d} |".format(num_val_gallery_pids, len(val_gallery)))
        logger.info("  query         | {:5d} | {:8d} |".format(num_test_query_pids, len(test_query)))
        logger.info("  gallery       | {:5d} | {:8d} |".format(num_test_gallery_pids, len(test_gallery)))
        logger.info("  --------------------------------------------")
        logger.info("  total         | {:5d} | {:8d} | ".format(num_total_pids, num_total_imgs))
        logger.info("  --------------------------------------------")

        # 格式对齐：属性赋值
        self.train = train
        self.val_query = val_query
        self.val_gallery = val_gallery
        self.query = test_query
        self.gallery = test_gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = len(clothes2label)
        self.num_test_query_pids = num_test_query_pids
        self.num_test_gallery_pids = num_test_gallery_pids
        self.pid2clothes = pid2clothes
        self.num_query_imgs = len(test_query)  # 修复这里的 AttributeError

    def _get_robust_key(self, path):
        """ 提取关键路径部分以匹配属性字典中的 Key """
        return osp.join(*path.replace('\\', '/').split('/')[-3:])

    def get_pid2label_and_clothes2label(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))
        img_paths.sort()

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            names = osp.basename(img_path).split('.')[0].split('_')
            clothes = names[0] + '_' + names[-1]
            pid = int(names[0])
            pid_container.add(pid)
            clothes_container.add(clothes)

        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_path in img_paths:
            names = osp.basename(img_path).split('.')[0].split('_')
            clothes = names[0] + '_' + names[-1]
            pid = pid2label[int(names[0])]
            clothes_id = clothes2label[clothes]
            pid2clothes[pid, clothes_id] = 1

        return pid2label, clothes2label, pid2clothes

    def _process_dir(self, dir_path, pid2label=None, clothes2label=None, relabel=False, recam=0):
        if 'query' in dir_path:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        else:
            img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))
        img_paths.sort()

        # 加载属性
        imgdir2attribute = {}
        meta_path = osp.join(self.dataset_dir, self.meta_dir)
        with open(meta_path, 'r') as f:
            for line in f:
                parts = line.split()
                imgdir_raw, attr_id, is_present = parts[0], parts[1], parts[2]
                key = self._get_robust_key(imgdir_raw)
                if key not in imgdir2attribute:
                    imgdir2attribute[key] = [0 for _ in range(self.meta_dims)]
                imgdir2attribute[key][int(attr_id)] = int(is_present)

        dataset = []
        pid_container = set()

        for ii, img_path in enumerate(img_paths):
            names = osp.basename(img_path).split('.')[0].split('_')
            clothes = names[0] + '_' + names[-1]
            pid = int(names[0])
            pid_container.add(pid)
            camid = int(recam + ii)

            if relabel and pid2label is not None:
                pid = pid2label[pid]

            if relabel and clothes2label is not None:
                clothes_id = clothes2label[clothes]
            else:
                clothes_id = pid  # 非训练集通常直接用 pid

            # 使用 robust key 解决 KeyError
            current_key = self._get_robust_key(img_path)
            attrs = imgdir2attribute.get(current_key, [0] * self.meta_dims)

            # 格式：(路径, 身份ID, 摄像机ID, 衣服ID, 属性列表)
            dataset.append((img_path, pid, camid, clothes_id, attrs))

        num_pids = len(pid_container)
        return dataset, num_pids

    def _check_before_run(self):
        check_list = [self.dataset_dir, self.train_dir, self.val_query_dir,
                      self.val_gallery_dir, self.test_query_dir, self.test_gallery_dir]
        for d in check_list:
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")


if __name__ == '__main__':
    # 这里的 root 指向包含 last 文件夹的目录
    dataset = LaST(root='/home/ykding/dataset')
    print(f"Query images: {dataset.num_query_imgs}")