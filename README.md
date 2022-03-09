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

The Peruvian Sign Language dataset was created by https://github.com/gissemari/PeruvianSignLanguage

To generate the pickle file with the shape of N x C x T x V, where N is the number of samples, C the number of coordinates, T the number of frames, and V the number of joints, run the following code

```bash
python3 create_dataset.py --addExtraJoint
```

## <strong>Training the Neural Network</strong>
1. Run the following code for training without wandb tracking.
```bash
python3 train.py
```