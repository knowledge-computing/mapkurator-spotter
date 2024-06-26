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

ROOT = ""
_PREDEFINED_SPLITS_TEXT = {
    "totaltext_train_palette": (ROOT + "SceneImages/totaltext/totaltext-train/train_images", 
                                ROOT + "SceneImages/totaltext/totaltext-train/train_poly_palette.json"),
    "totaltext_train": (ROOT + "SceneImages/totaltext/totaltext-train/train_images", 
                        ROOT + "SceneImages/totaltext/totaltext-train/train_poly.json"),
    "totaltext_test": (ROOT + "SceneImages/totaltext/totaltext-test/test_images", 
                       ROOT + "SceneImages/totaltext/totaltext-test/test_poly.json"),
    
    "icdar_train_palette": (ROOT + "SceneImages/icdar/icdar-train/train_images", 
                            ROOT + "SceneImages/icdar/icdar-train/train_poly_palette.json"),
    "icdar_train": (ROOT + "SceneImages/icdar/icdar-train/train_images", 
                    ROOT + "SceneImages/icdar/icdar-train/train_poly.json"),
    "icdar_test": (ROOT + "SceneImages/icdar/icdar-test/test_images", 
                   ROOT + "SceneImages/icdar/icdar-test/test_poly.json"),
    
    "mlt_train": (ROOT + "SceneImages/mlt/mlt-train/train_images", 
                  ROOT + "SceneImages/mlt/mlt-train/train_poly.json"),
    "mlt_train_palette": (ROOT + "SceneImages/mlt/mlt-train/train_images", 
                          ROOT + "SceneImages/mlt/mlt-train/train_poly_palette.json"),
    
    "textocr_train": (ROOT + "SceneImages/textocr/textocr-train/train_images",
                      ROOT + "SceneImages/textocr/textocr-train/train_poly.json"),
    "textocr_train_palette": (ROOT + "SceneImages/textocr/textocr-train/train_images", 
                              ROOT + "SceneImages/textocr/textocr-train/train_poly_palette.json"),
    
    "synthtext1_train": (ROOT + "SynthText/SynthText1/train_images", 
                         ROOT + "SynthText/SynthText1/train_poly.json"),
    "synthtext2_train": (ROOT + "SynthText/SynthText2/train_images", 
                         ROOT + "SynthText/SynthText2/train_poly.json"),    
    
    "synmap_osm_train": (ROOT + "SynthMap/en-tiles/synmap/osm/train_images",
                         ROOT + "SynthMap/en-tiles/synmap/osm/train_poly.json"),
    "synmap_skeleton_train": (ROOT + "SynthMap/en-tiles/synmap/skeleton/train_images",
                              ROOT + "SynthMap/en-tiles/synmap/skeleton/train_poly.json"),
    "synmap_osm_ms_train": (ROOT + "SynthMap/en-tiles-multiscale/synmap/osm/train_images",
                            ROOT + "SynthMap/en-tiles-multiscale/synmap/osm/train_poly.json"),
    "synmap_skeleton_ms_train": (ROOT + "SynthMap/en-tiles-multiscale/synmap/skeleton/train_images",
                                 ROOT + "SynthMap/en-tiles-multiscale/synmap/skeleton/train_poly.json"),
    
    "weinman_train_palette": (ROOT + "MapImages/weinman/weinman-train/train_images", 
                              ROOT + "MapImages/weinman/weinman-train/train_poly_palette.json"), 
    "weinman_test": (ROOT + "MapImages/weinman/weinman-test/test_images", 
                     ROOT + "MapImages/weinman/weinman-test/test_poly.json"), 
    
    "rumsey_train_palette": (ROOT + "MapImages/rumsey/rumsey-train/train_images", 
                             ROOT + "MapImages/rumsey/rumsey-train/train_poly_palette.json"), 
    "rumsey_test": (ROOT + "MapImages/rumsey/rumsey-test/test_images", 
                    ROOT + "MapImages/rumsey/rumsey-test/test_poly.json"),     
    
    # for coords
    "synthtext1_coord_train": (ROOT + "SynthText/coord/SynthText1/train_images", 
                               ROOT + "SynthText/coord/SynthText1/train_poly.json"),
    "synthtext2_coord_train": (ROOT + "SynthText/coord/SynthText2/train_images", 
                               ROOT + "SynthText/coord/SynthText2/train_poly.json"),    
    "map_coord_train": ("/home/yaoyi/lin00786/data/critical-mass/coords/train/train_images", 
                        "/home/yaoyi/lin00786/data/critical-mass/coords/train/train_poly.json"),
    "map_coord_train1": ("/home/yaoyi/lin00786/data/critical-mass/coords/train1/train_images", 
                         "/home/yaoyi/lin00786/data/critical-mass/coords/train1/train_poly.json"),
    
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
