import tensorflow as tf
from model import ESPCN
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("epoch", 15000, "Number of epoch")
flags.DEFINE_integer("image_size", 17, "The size of image input")
flags.DEFINE_integer("c_dim", 3, "The size of channel")
flags.DEFINE_boolean("is_train", False, "if the train")
flags.DEFINE_integer(
    "scale", 3, "the size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 14, "the size of stride")
flags.DEFINE_string("checkpoint_dir", "checkpoint",
                    "Name of checkpoint directory")
flags.DEFINE_float("learning_rate", 1e-5, "The learning rate")
flags.DEFINE_integer("batch_size", 128, "the size of batch")
flags.DEFINE_string("result_dir", "result", "Name of result directory")
flags.DEFINE_string("test_img", "Test/Set14/comic.bmp", "test_img")


def main(_):  # ?
    with tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
        espcn = ESPCN(sess,
                      image_size=FLAGS.image_size,
                      is_train=FLAGS.is_train,
                      scale=FLAGS.scale,
                      c_dim=FLAGS.c_dim,
                      batch_size=FLAGS.batch_size,
                      test_img=FLAGS.test_img,
                      )

        espcn.train(FLAGS)


if __name__ == '__main__':
    tf.compat.v1.app.run()  # parse the command argument , the call the main function
