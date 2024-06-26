import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import re
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator

import glob
import shutil
from shapely.geometry import Polygon, LinearRing
from adet.evaluation import text_eval_script
import zipfile
import pickle

from adet.evaluation.lexicon_procesor import LexiconMatcher

NULL_CHAR = u'口'

class TextEvaluator(DatasetEvaluator):
    """
    Evaluate text proposals and recognition.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._tasks = ("polygon", "recognition")
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            raise AttributeError(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
            )
        
        self.voc_size = cfg.MODEL.TRANSFORMER.VOC_SIZE
        self.use_customer_dictionary = cfg.MODEL.TRANSFORMER.CUSTOM_DICT
        self.use_polygon = cfg.MODEL.TRANSFORMER.USE_POLYGON
        self.text_eval_confidence = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST
        
        self.CTLABELS = []
        if not self.use_customer_dictionary:
            if self.voc_size == 96:
                self.CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
            elif self.voc_size == 14:
                self.CTLABELS = ["0","1","2","3","4","5","6","7","8","9","'","\"","°"]
                
        else:
            with open(self.use_customer_dictionary, 'r') as f: # load txt file
                for line in f.readlines():
                    self.CTLABELS.append(line.strip())
                
        if len(self.CTLABELS) == 0: return
        self._lexicon_matcher = LexiconMatcher(dataset_name, cfg.TEST.LEXICON_TYPE, cfg.TEST.USE_LEXICON, 
                                               self.CTLABELS + [NULL_CHAR],
                                               weighted_ed=cfg.TEST.WEIGHTED_EDIT_DIST)
        
        assert(int(self.voc_size - 1) == len(self.CTLABELS)), "voc_size is not matched dictionary size, got {} and {}.".format(int(self.voc_size - 1), len(self.CTLABELS))

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        root = ''
        if dataset_name == "totaltext_test":
            self._text_eval_gt_path = root + "SceneImages/totaltext/totaltext-test/gt.zip"
            self._word_spotting = True
        elif dataset_name == "icdar15_test":
            self._text_eval_gt_path = root + "SceneImages/icdar/icdar-test/gt.zip"
            self._word_spotting = False
        elif dataset_name == "weinman_test":
            self._text_eval_gt_path = root + "MapImages/weinman/weinman-test-nonum/gt.zip"
            self._word_spotting = False
        elif dataset_name == "rumsey_test":
            self._text_eval_gt_path = root + "MapImages/rumsey/rumsey-test-nonum/gt.zip"
            self._word_spotting = False
        elif dataset_name == "taiwan_test":
            self._text_eval_gt_path = root + "MapImages/taiwan-1904/test/gt.zip"
            self._word_spotting = False
        else:
            self._text_eval_gt_path = ""
        
    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for _input, _output in zip(inputs, outputs):
            prediction = {"image_id": _input["image_id"]}
            instances = _output["instances"].to(self._cpu_device)
            prediction["instances"] = self.instances_to_coco_json(instances, _input["image_id"])
            self._predictions.append(prediction)

    def to_eval_format(self, instances):
        txt_out = []
        for instance in instances:
            rec = instance['rec']
            pts = np.array(instance['poly']).reshape(-1, 2).astype(int).tolist()
            score = instance['score']
            
            if float(score) < self.text_eval_confidence:
                continue
            try:
                pgt = Polygon(pts)
                if not pgt.is_valid:
                    print('An invalid detection is removed ... ')
                    continue
            except Exception as e:
                print(e)
                print('An invalid detection is removed ... ')
                continue
                    
            pRing = LinearRing(pts)
            if pRing.is_ccw:
                pts.reverse()    
                
            out_str = ','.join([str(int(p)) for p in np.array(pts).reshape(-1).tolist()])
            out_str = out_str + ',####' + rec
            txt_out.append(out_str)

        return txt_out
    
    def evaluate_with_official_code(self, det_file, gt_file):
        return text_eval_script.text_eval_main(det_file, gt_file, is_word_spotting=self._word_spotting)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))
            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if not self._text_eval_gt_path:
            return {}
        
        temp_dir = "./temp_det_results/"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)

        for prediction in predictions:
            image_id = prediction['image_id']
            instances = prediction['instances']
            out_txt = self.to_eval_format(instances)
            txt_file = os.path.join(temp_dir, '{:07d}.txt'.format(image_id))
            with open(txt_file, 'w') as fout:
                for out_str in out_txt:
                    fout.writelines(out_str + '\n')
            
        def generate_zip(folder, zip_file):
            def zipdir(path, ziph):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if 'txt' in file:
                            ziph.write(os.path.join(root, file), arcname=file)

            zipf = zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED)
            zipdir(folder, zipf)
            zipf.close()

        generate_zip(temp_dir, 'det.zip')
        shutil.rmtree(temp_dir)
        text_result = self.evaluate_with_official_code('det.zip', self._text_eval_gt_path)
        os.remove('det.zip')
        
        # parse
        eval_results = OrderedDict()        
        template = "(\S+): (\S+): (\S+), (\S+): (\S+), (\S+): (\S+)"
        for task in ("e2e_method", "det_only_method"):
            result = text_result[task]
            groups = re.match(template, result).groups()
            eval_results[groups[0]] = {groups[i*2+1]: float(groups[(i+1)*2]) for i in range(3)}
        return eval_results


    def instances_to_coco_json(self, instances, img_id):
        num_instances = len(instances)
        if num_instances == 0:
            return []
    
        scores = instances.scores.tolist()
        if self.use_polygon:
            pnts = instances.polygons.numpy()
        else:
            pnts = instances.beziers.numpy()
        recs = instances.recs.numpy()
        rec_scores = instances.rec_scores.numpy()
    
        results = []
        for pnt, rec, score, rec_score in zip(pnts, recs, scores, rec_scores):
            poly = self.pnt_to_polygon(pnt)
            s = self.decode(rec)
            word = self._lexicon_matcher.find_match_word(s, img_id=str(img_id), scores=rec_score)
            if word is None: continue;
            result = {
                "image_id": img_id,
                "category_id": 1,
                "poly": poly,
                "rec": word,
                "score": score
            }
            results.append(result)
        return results


    def pnt_to_polygon(self, ctrl_pnt):
        if self.use_polygon:
            return ctrl_pnt.reshape(-1, 2).tolist()
        else:
            u = np.linspace(0, 1, 20)
            ctrl_pnt = ctrl_pnt.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
            points = np.outer((1 - u) ** 3, ctrl_pnt[:, 0]) \
                + np.outer(3 * u * ((1 - u) ** 2), ctrl_pnt[:, 1]) \
                + np.outer(3 * (u ** 2) * (1 - u), ctrl_pnt[:, 2]) \
                + np.outer(u ** 3, ctrl_pnt[:, 3])
            
            # convert points to polygon
            points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
            return points.tolist()
        
    
    def decode(self, rec):
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                s += self.CTLABELS[c]
            elif c == self.voc_size - 1:
                s += NULL_CHAR
        return s
            
