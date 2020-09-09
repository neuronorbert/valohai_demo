# https://colab.research.google.com/github/Tony607/detectron2_instance_segmentation_demo/blob/master/Detectron2_custom_coco_data_segmentation.ipynb#scrollTo=Lnkg1PByUjGQ

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

import os
import zipfile

INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')
WEIGHTS_PATH = os.path.join(INPUTS_DIR, 'weights/model_final_f10217.pkl')
data_path = os.path.join(INPUTS_DIR, 'data/data.zip')
print(os.listdir(INPUTS_DIR))


def dfs_walk(dir_path):
    for path, dirs, files in os.walk(dir_path):
        print(path)
        for f in files:
            print(f)


with zipfile.ZipFile(data_path, 'r') as zip_ref:
    zip_ref.extractall('.')

# dfs_walk('.')

register_coco_instances("fruits_nuts", {}, "./data/trainval.json", "./data/images")

fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")
dataset_dicts = DatasetCatalog.get("fruits_nuts")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("fruits_nuts",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
# initialize from model zoo
cfg.MODEL.WEIGHTS = WEIGHTS_PATH
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = 5    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (data, fig, hazelnut)

cfg.OUTPUT_DIR = '/valohai/outputs'
#os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
