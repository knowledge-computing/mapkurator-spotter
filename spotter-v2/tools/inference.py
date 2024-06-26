# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import logging

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg

import pandas as pd

# constants
WINDOW_NAME = "COCO detections"


# MAPKURATOR:
#     MAP_MAPKURATOR_SYSTEM_DIR: '/data/weiweidu/usc-umn-inferlink-ta1/system/mapkurator/mapkurator-system/'

#     MODEL_WEIGHT_PATH: '/data/weiweidu/mapkurator/spotter_v2/PALEJUN/weights/pretrain_en_synthtext_synthmap_tt_ic_mlt/model_final.pth'
#     MODEL_WEIGHT_PATH: '/data/weiweidu/mapkurator/spotter_v2/PALEJUN/weights/pretrain_coord_synthtext_map/model_final.pth'

#     INPUT_DIR_PATH: '/data/weiweidu/criticalmaas_data/hackathon2/mvtzinc_maps/cropped_images/10639_2_g1000_s500/'

#     TEXT_SPOTTING_MODEL_DIR: '/data/weiweidu/usc-umn-inferlink-ta1/system/mapkurator/spotter_v2/PALEJUN/'
#     SPOTTER_CONFIG: '/data/weiweidu/usc-umn-inferlink-ta1/system/mapkurator/spotter_v2/PALEJUN/configs/inference_en.yaml'
#     SPOTTER_CONFIG: '/data/weiweidu/usc-umn-inferlink-ta1/system/mapkurator/spotter_v2/PALEJUN/configs/inference_coord.yaml'

#     OUTPUT_FOLDER: '/data/weiweidu/criticalmaas_data/hackathon2/mapkurator_output/10639_2_g1000_s500/'

# Example: CUDA_VISIBLE_DEVICES=0 python tools/inference.py --config-file configs/inference_coord_tmp.yaml --output_json --input /home/yaoyi/shared/critical-maas/9month/evaluation_corner_crops --output /home/yaoyi/lin00786/work/critical-maas/4-text-spotting-coordinates/spotting-coords/9month_hackathon/spotting_output_evaluation_corner_crops_thre03_spotter_v2/'

# CUDA_VISIBLE_DEVICES=0 python tools/inference.py --config-file configs/inference_en_tmp.yaml --output_json --input /home/yaoyi/shared/critical-maas/9month/evaluation_corner_crops --output /home/yaoyi/lin00786/work/critical-maas/4-text-spotting-coordinates/spotting-coords/9month_hackathon/spotting_output_evaluation_en_thre03_spotter_v2/


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    #l=args.opts[0].split(' ')
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    # cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    # cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    print('*** parsing argument ***')
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs.yaml",
        metavar="FILE",
        help="path to config file",
    )
    # parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    # parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument("--output", help="A file or directory to save json output files")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--output_json", action="store_true", help="Save outputs to json instead of image")
    return parser


if __name__ == "__main__":
    print('*****')
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    logging.info(args.input)
    
    if args.input:
        
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        
        
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()

            assert args.output_json == True
            if args.output_json:
                # modified code block to save output to json
                predictions, poly_text_score_dict_list = demo.inference_on_image(img)

                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["instances"]), time.time() - start_time))

                # output json file instead of visualization
                if not os.path.exists(args.output):
                    os.makedirs(args.output)
                
                out_filename = os.path.join(args.output, os.path.basename(path).split('.')[0] + '.json')
                df = pd.DataFrame(poly_text_score_dict_list)
                df.to_json(out_filename)

            else:
                print("[WARNING] - Make sure `output_json` == True")
                
    else:
        print("[WARNING] - Input path does not exist")
                