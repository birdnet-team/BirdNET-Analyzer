import tensorflow as tf


# Base class to share common logic
class CombinedModelBase(tf.keras.Model):
    def __init__(self, pb_model, classifier, name):
        super().__init__(name=name)
        self.pb_model = pb_model  # Track the loaded SavedModel
        self.classifier = classifier

    def get_config(self):
        return super().get_config()


class CombinedModelReplaceWithSigmoid(CombinedModelBase):
    def __init__(self, pb_model, classifier):
        super().__init__(pb_model, classifier, name="combined_model_replace_sigmoid")

    def call(self, inputs):
        embeddings_output = self.pb_model.signatures["embeddings"](inputs)
        embeddings = list(embeddings_output.values())[0]
        output = self.classifier(embeddings)
        return tf.sigmoid(output)


class CombinedModelReplaceNoSigmoid(CombinedModelBase):
    def __init__(self, pb_model, classifier):
        super().__init__(pb_model, classifier, name="combined_model_replace_no_sigmoid")

    def call(self, inputs):
        embeddings_output = self.pb_model.signatures["embeddings"](inputs)
        embeddings = list(embeddings_output.values())[0]
        return self.classifier(embeddings)


class CombinedModelAppendWithSigmoid(CombinedModelBase):
    def __init__(self, pb_model, classifier):
        super().__init__(pb_model, classifier, name="combined_model_append_sigmoid")

    def call(self, inputs):
        embeddings_output = self.pb_model.signatures["embeddings"](inputs)
        embeddings = list(embeddings_output.values())[0]
        basic_output = self.pb_model.signatures["basic"](inputs)
        basic = list(basic_output.values())[0]
        classified = self.classifier(embeddings)
        output = tf.concat([basic, classified], axis=-1)
        return tf.sigmoid(output)


class CombinedModelAppendNoSigmoid(CombinedModelBase):
    def __init__(self, pb_model, classifier):
        super().__init__(pb_model, classifier, name="combined_model_append_no_sigmoid")

    def call(self, inputs):
        embeddings_output = self.pb_model.signatures["embeddings"](inputs)
        embeddings = list(embeddings_output.values())[0]
        basic_output = self.pb_model.signatures["basic"](inputs)
        basic = list(basic_output.values())[0]
        classified = self.classifier(embeddings)
        return tf.concat([basic, classified], axis=-1)
