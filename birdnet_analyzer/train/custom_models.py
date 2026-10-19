import tensorflow as tf


def base_signature(pb_model, version: str, output: str):
    """The SavedModel call that yields ``output`` ("embeddings" or "predictions").

    Returns ``(fn, input_key, output_key)``. The 2.4 export has one signature per
    output, the 3.0 export a single ``serving_default`` returning both.
    """
    if version == "2.4":
        if output == "embeddings":
            return pb_model.signatures["embeddings"], "inputs", "embeddings"

        return pb_model.signatures["basic"], "inputs", "scores"

    return pb_model.signatures["serving_default"], "x", output


def call_base(pb_model, version: str, output: str, inputs):
    fn, input_key, output_key = base_signature(pb_model, version, output)

    return fn(**{input_key: inputs})[output_key]


class CombinedModelBase(tf.keras.Model):  # ty:ignore[unresolved-attribute]
    def __init__(self, pb_model, classifier, version, base_outputs_probabilities, name):
        super().__init__(name=name)
        self.pb_model = pb_model
        self.classifier = classifier
        self.version = version
        self.base_outputs_probabilities = base_outputs_probabilities

    def get_config(self):
        return super().get_config()

    def _embeddings(self, inputs):
        return call_base(self.pb_model, self.version, "embeddings", inputs)

    def _base_probabilities(self, inputs):
        scores = call_base(self.pb_model, self.version, "predictions", inputs)

        return scores if self.base_outputs_probabilities else tf.sigmoid(scores)


class CombinedModelReplaceWithSigmoid(CombinedModelBase):
    def __init__(self, pb_model, classifier, version, base_outputs_probabilities):
        super().__init__(
            pb_model,
            classifier,
            version,
            base_outputs_probabilities,
            name="combined_model_replace_sigmoid",
        )

    def call(self, inputs):
        return tf.sigmoid(self.classifier(self._embeddings(inputs)))


class CombinedModelAppendWithSigmoid(CombinedModelBase):
    def __init__(self, pb_model, classifier, version, base_outputs_probabilities):
        super().__init__(
            pb_model,
            classifier,
            version,
            base_outputs_probabilities,
            name="combined_model_append_sigmoid",
        )

    def call(self, inputs):
        classified = tf.sigmoid(self.classifier(self._embeddings(inputs)))

        return tf.concat([self._base_probabilities(inputs), classified], axis=-1)
