# PSL-GAN
Generative Adversarial Graph Convolutional Networks for Peruvian Sign Language Synthesis.

## <strong>System requirements</strong>
* Python 3.8.11
* Create a new environment using `conda create -y -n psl-gan python=3.8.11`
* `N_CUDA` environment variable is defined to choose the GPU in case of having more than one

    ```bash
    pip3 install -r requirements.txt
    ```

## <strong>Dataset</strong>
This repository contains the implementantion of [Kinetic GAN](https://github.com/DegardinBruno/Kinetic-GAN) applied to Peruvian Sign Language. It is used as a data augmentation method.

The Peruvian Sign Language dataset was created by https://github.com/gissemari/PeruvianSignLanguage. This [folder](https://drive.google.com/drive/folders/1vGjDimIRA2DHTaLU84gnwbyvGsN8kEtZ?usp=sharing) contains the following data:
* `segmented_signs.zip`: zip file with the segmented signs in MP4 format
* `raw_data_cocopose.json`: located in zip file. Contains the landmarks got by [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody) pose estimation model
* `raw_data_mediapipe.json`: Contains the landmarks got by [MediaPipe](https://google.github.io/mediapipe/solutions/pose.html) pose estimation model (body and hands) 

To generate the pickle file with the shape of N x C x T x V, where N is the number of samples, C the number of coordinates, T the number of frames, and V the number of joints, run the following code

* 27 number of joints
```bash
python3 create_dataset.py --minframes <MIN_FRAMES> --mininstances <MIN_FRAMES>  --leftHandLandmarks --rightHandLandmarks --rawCocoDataset <PATH_JSON_COCOPOSE> --inputPath <PATH_SEGMENTED_SIGNS> --nLandmarks 27 --useCoco --rawDataset <PATH_JSON_MEDIAPIPE>
```

* 24 number of joints
```bash
python3 create_dataset.py --minframes <MIN_FRAMES> --mininstances <MIN_FRAMES>  --leftHandLandmarks --rightHandLandmarks --rawCocoDataset <PATH_JSON_COCOPOSE> --inputPath <PATH_SEGMENTED_SIGNS> --nLandmarks 24 --useCoco --rawDataset <PATH_JSON_MEDIAPIPE> --addExtraJoint
```


## <strong>Training the Neural Network</strong>
1. Run the following code for training without wandb tracking.
```bash
python3 train.py
```