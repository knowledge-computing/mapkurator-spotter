# mapkurator-spotter

This repository contains the implementations of the text spotter for scanned historical map images, spotter-v2, in the mapKurator System. Spotter-v2 is built upon [Deformable-DETR](https://arxiv.org/abs/2010.04159) and [TESTR](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Text_Spotting_Transformers_CVPR_2022_paper.html).

Please refer to the [mapKurator documentation](https://knowledge-computing.github.io/mapkurator-doc/#/docs/modules/spot) for details.

## Model weights 
All the model weights can be found [here](https://drive.google.com/drive/folders/1wkQWoqBpqTwOHVxXiemrsKOp4mYqQZ0d?usp=drive_link)

### Model weight for Spotting English Text

You can donwload the [model weight](https://drive.google.com/file/d/1agOzYbhZPDVR-nqRc31_S6xu8yR5G1KQ/view?usp=drive_link) and a [model card](https://docs.google.com/document/d/1CfTFbUIiY0jhs-AE8aT2PhtcjDi6skjNopS3v6YE05g/edit?usp=drive_link). The [configuration file](./spotter-v2/configs/inference_en_test.yaml) for inferencing is under ./configs.

### Model weight for Spotting Coordinates

You can donwload the [model weight](https://drive.google.com/file/d/12_tc3zOmzLPUU41g_fbPVMkvRkjbSsRf/view?usp=drive_link). The [configuration file](./spotter-v2/configs/inference_coord_test.yaml) for inferencing is under ./configs.

### Model weights for Multilingual Text Spotting

mapkurator-spotter supports multilingual text spotting. Please add the corresponding configuration file and weight to run the model. 

| Supported Language      | Config folder  | Model weight | Model card      | 
|------------|---------------|----------------|------------------------------------| 
| Russian | [config](https://drive.google.com/drive/folders/11vjSjULrWkct4VyhRy6wtez-spgeE9lx?usp=drive_link) | [weight](https://drive.google.com/file/d/16046LiHoaOZTFmdJWwljk5Djj4RwtbqQ/view?usp=drive_link)  | [model card](https://docs.google.com/document/d/11hKt2QohpPywqFrHv6_FFP-ZlVrFAeYZmzAhiDBpvP4/edit?usp=drive_link)
| Chinese + Japanese| [config](https://drive.google.com/drive/folders/1H-qX_xEosq2eb8hS5PiFhfIUmeaItiKU?usp=drive_link) | [weight](https://drive.google.com/file/d/1CfWBju-hlEUDsHbYunioZ9DDQOijMXFK/view?usp=drive_link)  | [model card](https://docs.google.com/document/d/1exYTkNmZB0mJ_PTg7AiPlcltqz0b5qyp9STKr8Za884/edit?usp=drive_link)
| Arabic |[config](https://drive.google.com/drive/folders/1iHcgZQxq_J3bs1_sBZbS0IgLemTJuU3z?usp=drive_link) | [weight](https://drive.google.com/file/d/1nbv8MFn2gUYqiTrdMFKxRfhtFYI4Axdy/view?usp=drive_link)  | [model card](https://docs.google.com/document/d/1z8b8H4M_lua_2UHPMVWUn7bbiVktLTCb3QGRL7zjlA4/edit?usp=drive_link)

## Acknowledgement

Thanks to [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) for a standardized training and inference framework, and [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) for the implementation of multi-scale deformable cross-attention, and  [TESTR](https://github.com/mlpc-ucsd/TESTR/tree/main) for the implementation of Deformable-DETR text spotter for scene images.
