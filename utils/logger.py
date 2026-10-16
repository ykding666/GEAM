import logging
import os
import sys
import os.path as osp


def setup_logger(name, save_dir, if_train, filename=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # setup_logger may be called more than once in notebooks or launchers.
    # Reusing old handlers would print every message multiple times.
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if filename is None:
            filename = "train_log.txt" if if_train else "test_log.txt"
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
