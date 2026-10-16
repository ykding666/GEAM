# The commands below use the local distributed/launch.py launcher.
# If the distributed directory is unavailable, replace:
#   python -u distributed/launch.py
# with:
#   python -u -m torch.distributed.launch

# LTCC training
CUDA_VISIBLE_DEVICES=0,1 python -u distributed/launch.py --nproc_per_node=2 --master_port=6666 train.py --config_file configs/ltcc/eva02_l_maskmeta_random.yml MODEL.DIST_TRAIN True

# LTCC inference
CUDA_VISIBLE_DEVICES=0,1 python -u distributed/launch.py --nproc_per_node=2 --master_port=6661 test.py --config_file configs/ltcc/eva02_l_maskmeta_random.yml TEST.WEIGHT '/data2/dyk/logs/2025-12/MADE/logs/ltcc/eva02_l_meta_best-515.pth'

# PRCC training
CUDA_VISIBLE_DEVICES=0,1 python -u distributed/launch.py --nproc_per_node=2 --master_port=6666 train.py --config_file configs/prcc/eva02_l_maskmeta_random.yml MODEL.DIST_TRAIN True

# PRCC inference
CUDA_VISIBLE_DEVICES=0,1 python -u distributed/launch.py --nproc_per_node=2 --master_port=6661 test.py --config_file configs/prcc/eva02_l_maskmeta_random.yml TEST.WEIGHT '/data2/dyk/logs/2025-12/MADE/logs/prcc/eva02_l_meta_best-660.pth'

# Celeb-reID-light training
CUDA_VISIBLE_DEVICES=0,1 python -u distributed/launch.py --nproc_per_node=2 --master_port=6666 train.py --config_file configs/Celeb_light/eva02_l_maskmeta_random.yml MODEL.DIST_TRAIN True

# Celeb-reID-light inference
CUDA_VISIBLE_DEVICES=0,1 python -u distributed/launch.py --nproc_per_node=2 --master_port=6661 test.py --config_file configs/Celeb_light/eva02_l_maskmeta_random.yml TEST.WEIGHT '/data2/dyk/logs/2025-12/MADE/logs/celeb_light/eva02_l_meta_best-743.pth'

# Celeb-reID training
CUDA_VISIBLE_DEVICES=0,1 python -u distributed/launch.py --nproc_per_node=2 --master_port=6666 train.py --config_file configs/Celeb/eva02_l_maskmeta_random.yml MODEL.DIST_TRAIN True

# Celeb-reID inference
CUDA_VISIBLE_DEVICES=0,1 python -u distributed/launch.py --nproc_per_node=2 --master_port=6661 test.py --config_file configs/Celeb/eva02_l_maskmeta_random.yml TEST.WEIGHT '/data2/dyk/logs/2025-12/MADE/logs/celeb/eva02_l_meta_best-791.pth'

# LaST training
CUDA_VISIBLE_DEVICES=0,1 python -u distributed/launch.py --nproc_per_node=2 --master_port=6666 train.py --config_file configs/last/eva02_l_maskmeta_random.yml MODEL.DIST_TRAIN True

# LaST inference
CUDA_VISIBLE_DEVICES=0,1 python -u distributed/launch.py --nproc_per_node=2 --master_port=6661 test.py --config_file configs/last/eva02_l_maskmeta_random.yml TEST.WEIGHT '/data2/dyk/logs/2025-12/MADE/logs/prcc/last/eva02_l_meta_best-818.pth'
