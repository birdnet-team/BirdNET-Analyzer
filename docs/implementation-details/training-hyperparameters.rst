Hyperparameters for Custom Classifier Training
================================================

This document provides detailed information on the hyperparameters used during the training of custom classifiers in the BirdNET-Analyzer.

The GUI always trains with the defaults given below. They can be changed on the command line (``birdnet_analyzer.train``) and in the Python API (``birdnet_analyzer.train.train``).

Epochs
----------

Determines the maximum number of times the entire training dataset is passed through the model. Defaults to 50.
If the validation performance does not improve for a certain number of epochs, training will stop before reaching the specified number of epochs.

Batch Size
----------------

Specifies the number of training samples processed at the same time. Defaults to 32.
This values is very dependent on the available hardware (RAM), if you are not sure, leave it at the default value.

Learning Rate
-------------------

Controls how much to change the model parameters in response to the estimated error each time the model weights are updated.
A smaller learning rate means the model learns more slowly but can lead to better convergence, while a higher learning rate speeds up training but may overshoot optimal solutions.

The default depends on the base model, because the embeddings of BirdNET 3.0 are on a much smaller scale than those of BirdNET 2.4: 0.1 for BirdNET 3.0 and 0.01 for BirdNET 2.4.
Values between 0.03 and 0.3 work about as well on BirdNET 3.0, values between 0.01 and 0.03 on BirdNET 2.4, where 0.1 makes the training unstable.
Much smaller values leave the classifier under-trained.

Hidden Units
-------------------

Specifies the number of hidden units in the classifier. If this is set to 0 (the default), a single layer classifier will be trained.
A high number of hidden units improves the model's ability to learn complex patterns but may lead to overfitting.

A classifier with hidden units needs a lower learning rate, such as 0.003: at the default learning rate of BirdNET 3.0 it often fails to train.

Dropout Rate
-------------------

Specify a rate to randomly disable a fraction of the hidden units during training to prevent overfitting. Defaults to 0.

Label Smoothing
---------------------

Label smoothing is a technique used to prevent the model from becoming overconfident by adjusting the target labels. Off by default.
It scales every label by (1 - alpha) for a small alpha (0.1) and spreads alpha evenly over all classes.

Upsampling
-------------------

Upsampling can be used to balance the dataset by using samples from underrepresented classes to create additional synthetic examples.
The upsampling factor determines the minimum number of samples for each class based on the number of samples in the largest class.
For example, if the largest class has 100 samples and the upsampling factor is set to 0.5, classes with fewer than 50 samples will be upsampled to have 50 samples.
Defaults to 0, no upsampling.

Upsampling makes a classifier report more: a detector for a single sound can find more of its calls but also flag more background noise.

Upsampling modes:
    - **Repeat**: Uses existing samples multiple times.
    - **Mean**: Creates new samples by averaging two random existing samples.
    - **Linear**: Creates new samples by interpolating between two random existing samples.
    - **SMOTE**: SMOTE (Synthetic Minority Over-sampling Technique) is similar to linear interpolation but creates synthetic samples between a random sample and one of its nearest neighbors.

Mixup
-------------------

Mixup is a data augmentation technique that replaces existing training samples and their labels with a randomly weighted combination between itself and another random sample. Off by default.
This can help improve the model's robustness and generalization as the combination is not limited to samples from the same class.

Focal loss
-------------------

Focal loss is a loss function that helps the model focus on hard-to-classify examples by down-weighting the loss contribution from easy-to-classify examples. Off by default.

Autotune
-------------------

Autotune (``--autotune``) runs multiple training runs (aka trials) with different hyperparameter settings and selects the best performing settings based on the validation data. Off by default.
Its first trial uses the given settings. The parameters used for training the final classifier are saved alongside the resulting classifier.
You can specify the number of trials, the number of folds and the number of repeats.

Each trial is scored on the validation split of the training data. With more than one fold, each trial is trained once per fold instead and scored by the mean over the folds.
That makes the scores steadier, but multiplies the tuning time by roughly the number of folds.
Folds are mainly worth it for small datasets of a few hundred samples, where the validation split holds only a handful of examples per class.
From about 1,000 samples on, keep 1 fold and spend the time on more trials instead. Repeats run the folds again on a new split and cost as much time again.
