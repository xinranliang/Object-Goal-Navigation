# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import pickle

# import some common detectron2 utilities
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# constants
scenes = {}
scenes["train"] = [
    'Allensville',
    'Beechwood',
    'Benevolence',
    'Coffeen',
    'Cosmos',
    'Forkland',
    'Hanson',
    'Hiteman',
    'Klickitat',
    'Lakeville',
    'Leonardo',
    'Lindenwood',
    'Marstons',
    'Merom',
    'Mifflinburg',
    'Newfields',
    'Onaga',
    'Pinesdale',
    'Pomaria',
    'Ranchester',
    'Shelbyville',
    'Stockman',
    'Tolstoy',
    'Wainscott',
    'Woodbine',
]

scenes["val"] = [
    'Collierville',
    'Corozal',
    'Darden',
    'Markleeville',
    'Wiconisco',
]

coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    # "dining-table": 6,
    # "oven": 7,
    # "sink": 8,
    # "refrigerator": 9,
    # "book": 10,
    # "clock": 11,
    # "vase": 12,
    # "cup": 13,
    # "bottle": 14
}

coco_categories_mapping = {
    0: 56,  # chair
    1: 57,  # couch
    2: 58,  # potted plant
    3: 59,  # bed
    4: 61,  # toilet
    5: 62,  # tv
    # 60: 6,  # dining-table
    # 69: 7,  # oven
    # 71: 8,  # sink
    # 72: 9,  # refrigerator
    # 73: 10,  # book
    # 74: 11,  # clock
    # 75: 12,  # vase
    # 41: 13,  # cup
    # 39: 14,  # bottle
}

coco_categories_objects = ["chair", "couch", "potted plant", "bed", "toilet", "tv"]


master_scene_dir = "/home/xinranliang/projects/interactive-robustness/sem-exp/data/scene_datasets/gibson_semantics/"
master_save_dir = '/home/xinranliang/projects/sem-exp/logs/2023-03-04/eval_pretrain/rollouts/'
master_model_dir = '/home/xinranliang/projects/interactive-robustness/seal/models/'

def get_all_coco_categories():
    cfg = get_cfg()
    cfg.merge_from_file(master_model_dir + "configs/mask_rcnn_R_50_FPN_3x.yaml")
    object_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    return object_classes

def get_habitat_dicts(scene_str, max_frames=1000):
    dataset_dicts = []
    for idx in range(max_frames):
        with open(master_save_dir + "%s/dict/%s_%03d.pkl" % (scene_str, scene_str, idx), 'rb') as f:
            dict_obj = pickle.load(f)
            for single_object in dict_obj['annotations']:
                simple_catid = single_object['category_id']
                single_object['category_id'] = coco_categories_mapping[simple_catid]
        dataset_dicts.append(dict_obj)
    return dataset_dicts

def main_test():
    coco_objects = get_all_coco_categories()
    for scene_name in scenes['val']:
        DatasetCatalog.register("habitat_" + scene_name, lambda scene_name=scene_name: get_habitat_dicts(scene_name))
        MetadataCatalog.get("habitat_" + scene_name).set(thing_classes=coco_objects)
    
    for scene_name in scenes['val']:
        habitat_metadata = MetadataCatalog.get("haitat_" + scene_name)
        dataset_dicts = get_habitat_dicts(scene_name)
        for d in dataset_dicts:
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=habitat_metadata, scale=1)
            out = visualizer.draw_dataset_dict(d)
            os.makedirs(master_save_dir + scene_name + '/label/', exist_ok=True)
            cv2.imwrite(master_save_dir + scene_name + '/label/' + d['image_id'].split('_')[1] + '.png', out.get_image()[:, :, ::-1])

def main_train():
    coco_objects = get_all_coco_categories()
    for scene_name in scenes['train']:
        DatasetCatalog.register("habitat_" + scene_name, lambda scene_name=scene_name: get_habitat_dicts(scene_name))
        MetadataCatalog.get("habitat_" + scene_name).set(thing_classes=coco_objects)
    
    for scene_name in scenes['train']:
        habitat_metadata = MetadataCatalog.get("haitat_" + scene_name)
        dataset_dicts = get_habitat_dicts(scene_name)
        for d in dataset_dicts:
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=habitat_metadata, scale=1)
            out = visualizer.draw_dataset_dict(d)
            os.makedirs(master_save_dir + scene_name + '/label/', exist_ok=True)
            cv2.imwrite(master_save_dir + scene_name + '/label/' + d['image_id'].split('_')[1] + '.png', out.get_image()[:, :, ::-1])

def main_small():
    coco_objects = get_all_coco_categories()

    for scene_name in ["Allensville", "Forkland"]:
        DatasetCatalog.register("habitat_" + scene_name, lambda scene_name=scene_name: get_habitat_dicts(scene_name, max_frames=40))
        MetadataCatalog.get("habitat_" + scene_name).set(thing_classes=coco_objects)
    
    for scene_name in ["Allensville", "Forkland"]:
        habitat_metadata = MetadataCatalog.get("haitat_" + scene_name)
        dataset_dicts = get_habitat_dicts(scene_name, max_frames=40)
        for d in dataset_dicts:
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=habitat_metadata, scale=1)
            out = visualizer.draw_dataset_dict(d)
            os.makedirs(master_save_dir + scene_name + '/label/', exist_ok=True)
            cv2.imwrite(master_save_dir + scene_name + '/label/' + d['image_id'].split('_')[1] + '.png', out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process groundtruth label visualization.')
    parser.add_argument('--mode', type=str, help='mode: train or test')
    args = parser.parse_args()
    
    if args.mode == 'train':
        main_train()
    elif args.mode == 'test':
        main_test()
    else:
        main_small()