from abstract_loss import Loss
import tensorflow as tf

class TfGram(Loss):

    @staticmethod
    def calculate_loss(input_tensor: tf.Tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)