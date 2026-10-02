**Deep Learning-Based Coarse Alignment and Cophasing of a Distributed-Aperture Telescope. Application to the Small ExoLife Finder (SELF)**



This repository code used for the training and evaluation of several Convolutional Neural Network (CNN) models, including the noise-robustness analysis, together with the associated test data.

These CNN architectures are trained for predicting the perturbations applied to the M1 mirrors of the simplified configuration of SELF (four M1-M2 mirror pairs) from point-spread-function (PSF) data.

The available architectures are:

•	Custom CNN

•	ResNet18

•	EfficientNetB0

•	MobileNetV3

•	VGG16

All architectures are adapted for single-channel PSF images and a 12-dimensional regression output. The architectures are defined in models.py.



**Requirements**



The code was developed and tested with:

•	Python 3.9

•	PyTorch 2.1

•	Torchvision 0.16

•	CUDA 11.8 (for GPU execution)

The complete software environment is specified in: environment.yml



**Creating the environment**



Install Anaconda or Miniconda and create the environment with:

conda env create -f environment.yml

Activate it with:

conda activate zemax-cnn

The code automatically selects a CUDA GPU when available and otherwise uses the CPU.



**Input data**

A small sample dataset is included in this repository for testing and demonstration purposes. The full dataset is not publicly distributed due to data access restrictions.



A larger test dataset containing approximately 1,000 samples may be provided upon reasonable request, subject to applicable data access and usage conditions. For access requests, please contact the corresponding author of the associated publication.



The scripts read PSF data stored in Feather files. The expected input contains:

psf\_flat — flattened PSF image

c1\_m1tx

c1\_m1ty

c2\_m1tx

c2\_m1ty

c3\_m1tx

c3\_m1ty

c4\_m1tx

c4\_m1ty

c1\_m1p

c2\_m1p

c3\_m1p

c4\_m1p

The target vector therefore contains 12 regression outputs:

•	four M1 tilt-X values,

•	four M1 tilt-Y values,

•	four M1 piston values.

The repository includes the test data used for evaluation.



**Training and testing**



The main training and evaluation workflow is implemented in:

code/train\_and\_test.py

Run it from the repository root:

python code/train\_and\_test.py



**Train/test Split**



The train/test partition is saved as:

output/data\_split.npz

The split can be generated once and subsequently reused when training the different architectures. This allows different neural-network architectures to be evaluated using exactly the same train/test samples.



**Trained models**



For each architecture, trained model weights are stored in the corresponding output directory.

For example:

output/

└── Custom\_CNN/

&#x20;   └── saved\_models/

&#x20;       └── best\_model\_fold9.pt

The corresponding StandardScaler.pkl file contains the target-data scaling used during training and must be used when making predictions with the trained model.



The repository includes the model definitions used for the experiments rather than relying only on the names of standard architectures. The trained model weights and corresponding StandardScaler.pkl files are provided so that the supplied test data can be evaluated without retraining the networks.



The saved data\_split.npz file allows the same train/test partition to be reused when comparing different model architectures.



Because Gaussian noise is randomly generated during the noise-robustness evaluation, individual noisy realizations may differ between executions unless the NumPy random seed is explicitly fixed.



**Noise robustness evaluation**



The script:

code/noise\_robustness.py

evaluates the trained models in the presence of additive Gaussian noise.



The evaluated SNR values are:

10, 15, 20, 25, 30, 35, 40, 45, 50 dB



Run the evaluation with:

python code/noise\_robustness.py



The model to evaluate is selected using:

MODEL\_NAME = "Custom CNN"

and the corresponding model and scaler paths are specified in MODEL\_REGISTRY.

MODEL\_REGISTRY = {

&#x20;   "Custom CNN": {

&#x20;       "model\_path": "./output/CustomCNN/saved\_models/best\_model\_fold1.pt",

&#x20;       "scaler\_path": "./output/CustomCNN/StandardScaler.pkl",

&#x20;   },

}



Change MODEL\_NAME and MODEL\_REGISTRY to evaluate another architecture.



The noise evaluation does not resample the test dataset. The same test samples are evaluated at every SNR level, with newly generated Gaussian noise for each evaluation.



**Noise evaluation metrics**



For each SNR level, the script calculates the RMSE separately for:

•	tilt-X,

•	tilt-Y,

•	piston.

For each quantity, the RMSE is calculated independently for the four mirrors and then averaged over the four mirrors. The corresponding standard deviation across the four mirror RMSE values is also stored.



The output is written to:

output/noise\_robustness/

Both JSON and CSV formats are produced.



**Reproducing the results**



A typical workflow is:

1\. Create the environment

conda env create -f environment.yml

conda activate zemax-cnn

2\. Place the supplied data in the expected directory

The Feather files should be placed in the directory specified by DATA\_DIR in the corresponding script.

3\. Train the models

python code/train\_and\_test.py

The first execution creates:

output/data\_split.npz

Subsequent model training runs reuse this same partition.

4\. Evaluate noise robustness

After the trained model and scaler have been generated:

python code/noise\_robustness.py

Set MODEL\_NAME and the paths in MODEL\_REGISTRY to the model that is being evaluated.



**Citation**



If you use this code or the associated test data, please cite the paper:

https://doi.org/10.48550/arXiv.2608.25173

The repository associated with the paper is:

https://github.com/LIOM-IAC/SELF\_M1-A-CNN-based-coarse-alignment-and-cophasing

