# mapkurator-spotter

This repository contains the implementations of the text spotter, Spotter-v2, in the mapKurator System. Spotter-v2 is built upon [Deformable-DETR](https://arxiv.org/abs/2010.04159) and [TESTR](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Text_Spotting_Transformers_CVPR_2022_paper.html).

Spotter-v2 adopts a novel feature sampling strategy that samples relevant image features around the target points for predicting boundary points, which leads to enhanced detection and recognition performance.

Please refer to the [mapKurator documentation](https://knowledge-computing.github.io/mapkurator-doc/#/docs/modules/spot) for details.


## model weights 

You can donwload Spotter-v2 (English) model weight [here](https://drive.google.com/file/d/1agOzYbhZPDVR-nqRc31_S6xu8yR5G1KQ/view?usp=drive_link) and a [model card](https://docs.google.com/document/d/1CfTFbUIiY0jhs-AE8aT2PhtcjDi6skjNopS3v6YE05g/edit?usp=drive_link) 

## Acknowledgement

Thanks to [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) for a standardized training and inference framework, and [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) for the implementation of multi-scale deformable cross-attention, and  [TESTR](https://github.com/mlpc-ucsd/TESTR/tree/main)  for the implementation of Deformable-DETR text spotter for scene images.
