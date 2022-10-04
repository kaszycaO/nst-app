from abstract_loss import Loss
import tensorflow as tf

class CustomGram(Loss):

    @staticmethod
    def calculate_loss(input_tensor: tf.Tensor):
        channels = int(input_tensor.shape[-1])
        feature_map = tf.reshape(input_tensor, [-1, channels])  # vectorize
        gram = tf.matmul(feature_map, feature_map, transpose_a=True)
        return gram