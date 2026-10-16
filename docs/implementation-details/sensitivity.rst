Sensitivity
===============================

Starting with version 2.0.0, the BirdNET-Analyzer uses a new sensitivity implementation.
Our goal with this change is to make the sensitivity more intuitive by changing it so that higher sensitivity values always lead to higher confidence scores.

Comparison to the previous implementation
--------------------------------------------------------

Where the old implementation changed the slope of the sigmoid function, the new implementation performs a shift of the function.

The shift is applied to the model's logits. Models that apply the sigmoid inside their graph (BirdNET 3.0, classifiers trained on it, and Raven exports) have their logits recovered from the scores first, so the sensitivity behaves the same for every BirdNET model.

We have a little demo where you can explore the behavior of the old and the new sensitivity implementations `here <../_static/birdnet_sigmoid_sensitivity_old_vs_new.html>`_.