
import tensorflow as tf

@tf.function()
def train_step(image, nst, opt):
    total_variation_weight = 60
    with tf.GradientTape() as tape:
        outputs = nst.process_input(image)
        loss = nst.calculate_loss(outputs)
        loss += total_variation_weight*tf.image.total_variation(image)  # renoising

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

    return loss