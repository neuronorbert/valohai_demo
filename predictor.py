# https://colab.research.google.com/github/Tony607/detectron2_instance_segmentation_demo/blob/master/Detectron2_custom_coco_data_segmentation.ipynb#scrollTo=Lnkg1PByUjGQ

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.events import EventWriter, get_event_storage

from PIL import Image
from werkzeug.debug import DebuggedApplication
from werkzeug.wrappers import Request, Response


import os
import zipfile
import json

predictor = None


def init_predictor():
    # INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')
    WEIGHTS_PATH = 'model_final.pth'  # os.path.join(INPUTS_DIR, 'weights/model_final_f10217.pkl')

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 2
    # initialize from model zoo
    cfg.MODEL.WEIGHTS = WEIGHTS_PATH
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (data, fig, hazelnut)

    return DefaultPredictor(cfg)


def read_image_from_wsgi_request(environ):
    request = Request(environ)
    if not request.files:
        return None
    file_key = list(request.files.keys())[0]
    file = request.files.get(file_key)
    img = Image.open(file.stream)
    img.load()
    return img


def predict_wsgi(environ, start_response):
    img = read_image_from_wsgi_request(environ)
    if not img:
        return Response('no file uploaded', 400)(environ, start_response)

    global predictor
    if not predictor:
        predictor = init_predictor()
    prediction = predictor(img)
    response = Response(json.dumps(prediction), mimetype='application/json')
    return response(environ, start_response)


predict_wsgi = DebuggedApplication(predict_wsgi)

if __name__ == '__main__':
    from werkzeug.serving import run_simple

    run_simple('localhost', 3000, predict_wsgi)
