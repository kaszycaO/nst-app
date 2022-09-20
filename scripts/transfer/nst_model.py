import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import logging


class NstModel:

  def __init__(
    self,
    content_img,
    style_img,
    networks,
    mode,
    gram_matrix_type="custom",
    alpha=1e-2,
    beta=1e4
  ):
 
    """ Nst model

    :param content_image: Image with content
    :type content_image: tensorflow.python.framework.ops.EagerTensor
    :param style_image: Image with style
    :type style_image: tensorflow.python.framework.ops.EagerTensor
    :param networks: Dictionary with style and content networks 
                    (see Przygotowanie sieci)
    :type networks: dict
    :param gram_matrix_type: Gram matrix implemetation type, custom or tf, 
                             default custom
    :type gram_matrix_type: str
    """
    

    self.style = networks["style"]
    self.content = networks["content"]

    self.alpha = alpha
    self.beta = beta

    logging.info(
      f"Style: {self.style['net'].name} content: {self.content['net'].name}"
    )
    logging.info(
      f"Params: {type(content_img)}, {type(style_img)}, {networks}, {mode}, {gram_matrix_type}"
    )
    logging.info(
      f"Alpha: {alpha}, Beta: {beta}"
    )

    self.num_content_layers = len(self.content["clayers"])
    self.num_style_layers = len(self.style["slayers"])

    self.content_layers = self.content["clayers"]
    self.style_layers = self.style["slayers"]

    self.style_model = self.get_model(self.style["net"], self.style_layers)
    self.content_model = self.get_model(self.content["net"], self.content_layers)

    self.mode = mode

    if gram_matrix_type == "custom":
        self.gram_matrix = self.custom_gram_matrix
    elif gram_matrix_type == "tf":
        self.gram_matrix = self.tensorflow_gram_matrix
    else:
        raise Exception(f"Invalid gram matrix type: {gram_matrix_type}" + 
                        "Available options are: custom, tf")

    # Get feature maps
    self.style_features = self.process_input(style_img)["style"]
    self.content_features = self.process_input(content_img)["content"]


  def get_model(self, network, names):
    outputs = [network.get_layer(name).output for name in names]
    model = tf.keras.Model([network.input], outputs)
    model.trainable = False
    return model

  def process_input(self, input_img):
    if "VGG" in self.mode:
        input_img = input_img * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(input_img)
    else:
        preprocessed_input = tf.image.per_image_standardization(input_img)
      
    content_outputs = self.content_model(preprocessed_input)
    style_outputs = self.style_model(preprocessed_input)
    style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]
    style_dict = {style_name: value for style_name, value in 
                  zip(self.style_layers, style_outputs)}

    if self.num_content_layers > 1:
      content_dict = {content_name: value for content_name, value in 
                    zip(self.content_layers, content_outputs)}
    else:
      content_dict = {self.content["clayers"][0]: content_outputs}

    return {'content': content_dict, 'style': style_dict}

  def custom_gram_matrix(self, input_tensor):
    channels = int(input_tensor.shape[-1])
    feature_map = tf.reshape(input_tensor, [-1, channels])  # vectorize
    gram = tf.matmul(feature_map, feature_map, transpose_a=True)
    return gram

  def tensorflow_gram_matrix(self, input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


  def calculate_loss(self, outputs):

    style_weight = self.alpha
    content_weight = self.beta
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    style_loss = tf.reduce_sum([1.0 / self.num_style_layers * \
                                tf.reduce_mean((style_outputs[name]-self.style_features[name])**2) 
                                for name in style_outputs.keys()])
    
    style_loss *= style_weight

    content_loss = tf.reduce_sum([tf.reduce_mean((content_outputs[name]-self.content_features[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / self.num_content_layers

    loss = style_loss + content_loss
    return loss