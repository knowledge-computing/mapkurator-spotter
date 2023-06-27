import os

from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .datasets.text import register_text_instances

# register plane reconstruction

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}

metadata_pic = {
    "thing_classes": ["person"]
}

ROOT = "./spotter-data/"
SYNMAP_ROOT = "./spotter-data/"
_PREDEFINED_SPLITS_TEXT = {
    "syntext1_poly_train": (ROOT + "syntext1/images", ROOT + "syntext1/train_poly.json"),
    "syntext2_poly_train": (ROOT + "syntext2/images", ROOT + "syntext2/train_poly.json"),
    "synthtext1_poly_train": (ROOT + "SynthText1/train_images", ROOT + "SynthText1/train_poly.json"),
    "synthtext2_poly_train": (ROOT + "SynthText2/train_images", ROOT + "SynthText2/train_poly.json"),    
    "synmap_osm_train": (SYNMAP_ROOT + "en-tiles/synmap/osm/train_images",
                         SYNMAP_ROOT + "en-tiles/synmap/osm/train_poly.json"),
    "synmap_skeleton_train": (SYNMAP_ROOT + "en-tiles/synmap/skeleton/train_images",
                              SYNMAP_ROOT + "en-tiles/synmap/skeleton/train_poly.json"),
    "synmap_osm_ms_train": (SYNMAP_ROOT + "en-tiles-multiscale/synmap/osm/train_images",
                         SYNMAP_ROOT + "en-tiles-multiscale/synmap/osm/train_poly.json"),
    "synmap_skeleton_ms_train": (SYNMAP_ROOT + "en-tiles-multiscale/synmap/skeleton/train_images",
                              SYNMAP_ROOT + "en-tiles-multiscale/synmap/skeleton/train_poly.json"),
}


metadata_text = {
    "thing_classes": ["text"]
}


def register_all_coco(root="datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata_pic,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_text_instances(
            key,
            metadata_text,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
        
register_all_coco()
