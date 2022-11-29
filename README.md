# Grand Challenge: RAVIR

## About the challenge

This section contains information from the official RAVIR Grand Challenge [website](https://ravir.grand-challenge.org/RAVIR/).

<details><summary>Click to expand</summary>
<p>

--------------------

### RAVIR: A Dataset and Methodology for the Semantic Segmentation and Quantitative Analysis of Retinal Arteries and Veins in Infrared Reflectance Imaging

The retinal vasculature provides important clues in the diagnosis and monitoring of systemic diseases including hypertension and diabetes. The microvascular system is of primary involvement in such conditions, and the retina is the only anatomical site where the microvasculature can be directly observed. The objective assessment of retinal vessels has long been considered a surrogate biomarker for systemic vascular diseases, and with recent advancements in retinal imaging and computer vision technologies, this topic has become the subject of renewed attention. In this paper, we present a novel dataset, dubbed RAVIR, for the semantic segmentation of Retinal Arteries and Veins in Infrared Reflectance (IR) imaging. It enables the creation of deep learning-based models that distinguish extracted vessel type without extensive post-processing.

For more information, please refer to the dataset paper:

[RAVIR: A Dataset and Methodology for the Semantic Segmentation and Quantitative Analysis of Retinal Arteries and Veins in Infrared Reflectance Imaging](https://arxiv.org/pdf/2203.14928.pdf)

### Test Set Evaluation

Note that the following needs to be taken into account when preparing the prediction files:
1. Predictions should be a PNG file as __2D maps__ containing artery and vein classes with size (__768,768__) [submitting maps with size (768,768,3) will trigger an error in evaluation).
2. Artery and vein classes should have labels of __128__ and __256__ respectively. Background should have a label of __0__. 
3. The filenames must exactly match the name of the images in the test set [__IR_Case_006.png, ..., IR_Case_060.png__].

Once segmentation outputs are obtained in the correct format, participants should place the them in a folder named test and submit a zipped version of this folder to the server. 

The Dice and Jaccard scores will be calculated for every image in the test set for both artery and vein classes. The leaderbord is sorted on the basis of the best average Dice score.

### Data Description

The images in RAVIR dataset were captured using infrared (815nm) Scanning Laser Ophthalmoscopy (SLO), which in addition to having higher quality and contrast, is less affected by opacities in optical media and pupil size. RAVIR images are sized at 768 × 768, captured using a Heidelberg Spectralis camera with a 30° FOV and compressed in the Portable Network Graphics (PNG) format. Each pixel in the images has a reference length of 12.5 microns. RAVIR project was carried out with the approval of the Institutional Review Board at UCLA and adhered to the tenets of the Declaration of Helsinki.

### Data Usage Agreement

RAVIR dataset is distributed under [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) Licence and may only be used for non-commercial purposes. The following manuscripts must be cited if RAVIR dataset is used in any instances:

[1]: Hatamizadeh, A., Hosseini, H., Patel, N., Choi, J., Pole, C., Hoeferlin, C., Schwartz, S. and Terzopoulos, D., 2022. [RAVIR: A Dataset and Methodology for the Semantic Segmentation and Quantitative Analysis of Retinal Arteries and Veins in Infrared Reflectance Imaging](https://ieeexplore.ieee.org/abstract/document/9744459?casa_token=bVo3jBzkoxYAAAAA:t2O9G3Y6cK05AxFEBL8oi_PyrOzbypsBHNCiKuqkiYTNG3gwzy7oNVo3dPpSmxFS-9B2dZJJzmjP1Q). IEEE Journal of Biomedical and Health Informatics.

[2]: Hatamizadeh, A., 2020. [An Artificial Intelligence Framework for the Automated Segmentation and Quantitative Analysis of Retinal Vasculature. University of California](https://escholarship.org/uc/item/4r63v2bd), Los Angeles.

--------------------

</p>
</details>

## Authors

| Name | Nickname | Team |
| --- | --- | --- |
| Miranda Gisudden | [aphrenia](https://github.com/aphrenia) | KTH-CBH-CM2003-1 |
| Jonas Stylbäck | [Stylback](https://github.com/Stylback/) | KTH-CBH-CM2003-1 |

## Running instructions

1. Install [Jupyter Notebook](https://jupyter.org/install)
2. Download the `code` and `dataset` folders
3. Open `prerequisite.ipynb` to read about the prerequisites and install dependencies
4. Run the pipeline in `pipeline.ipynb`

## Main findings

### Pipeline

NOTE: Preliminary pipeline

![pipeline diagram](https://github.com/Stylback/ravir-challenge/blob/main/media/pipeline.png?raw=true)

## License and usage

Documentation and findings: TBD

Source code: TBD

Data usage: To use the RAVIR dataset you need to agree to their Data Usage Agreement, please see the [official website](https://ravir.grand-challenge.org/data/) for license and information.