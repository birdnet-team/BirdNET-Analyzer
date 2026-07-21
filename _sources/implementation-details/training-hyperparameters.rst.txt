Hyperparameters for Custom Classifier Training
================================================

This document provides detailed information on the hyperparameters used during the training of custom classifiers in the BirdNET-Analyzer.

Epochs
----------

Determines the maximum number of times the entire training dataset is passed through the model.
If the validation performance does not improve for a certain number of epochs, training will stop before reaching the specified number of epochs.

Batch Size
----------------

Specifies the number of training samples processed at the same time.
This values is very dependent on the available hardware (RAM/VRAM), if you are not sure, leave it at the default value.

Learning Rate
-------------------

Controls how much to change the model parameters in response to the estimated error each time the model weights are updated.
A smaller learning rate means the model learns more slowly but can lead to better convergence, while a higher learning rate speeds up training but may overshoot optimal solutions.

Hidden Units
-------------------

Specifies the number of hidden units in the classifier. If this is set to 0, a single layer classifier will be trained.
A high number of hidden units improves the model's ability to learn complex patterns but may lead to overfitting.

Dropout Rate
-------------------

Specify a rate to randomly disable a fraction of the hidden units during training to prevent overfitting.

Label Smoothing
---------------------

Label smoothing is a technique used to prevent the model from becoming overconfident by adjusting the target labels.
It subtracts a small value (alpha) from the correct label and distributes it among the other labels.

Upsampling
-------------------

Upsampling can be used to balance the dataset by using samples from underrepresented classes to create additional synthetic examples.
The upsampling factor determines the minimum number of samples for each class based on the number of samples in the largest class.
For example, if the largest class has 100 samples and the upsampling factor is set to 0.5, classes with fewer than 50 samples will be upsampled to have 50 samples.

Upsampling modes:
    - **Repeat**: Uses existing samples multiple times.
    - **Mean**: Creates new samples by averaging two random existing samples.
    - **Linear**: Creates new samples by interpolating between two random existing samples.
    - **SMOTE**: SMOTE (Synthetic Minority Over-sampling Technique) is similar to linear interpolation but creates synthetic samples between a random sample and one of its nearest neighbors.

Mixup
-------------------

Mixup is a data augmentation technique that replaces existing training samples and their labels with a randomly weighted combination between itself and another random sample.
This can help improve the model's robustness and generalization as the combination is not limited to samples from the same class.

Focal loss
-------------------

Focal loss is a loss function that helps the model focus on hard-to-classify examples by down-weighting the loss contribution from easy-to-classify examples.