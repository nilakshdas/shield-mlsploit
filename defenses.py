from collections import OrderedDict

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from skimage.restoration import denoise_tv_bregman
import tensorflow as tf


def _get_image_from_arr(img_arr):
    return Image.fromarray(np.asarray(img_arr, dtype="uint8"))


class Defense(object):
    doctxt = ""
    option_doctxts = {}

    @staticmethod
    def apply(image, **kwargs):
        raise NotImplementedError


class JPEGDefense(Defense):
    doctxt = "This applies JPEG compression to an image to remove high frequency noise."
    option_doctxts = {
        "quality": (
            "This is the quality (0 - 100) of the output image. "
            "Lower quality means more compression is applied."
        )
    }

    @staticmethod
    def apply(image, quality=75):
        image = np.array(image, dtype=np.uint8)

        x = tf.constant(image)
        x = tf.image.decode_jpeg(
            tf.image.encode_jpeg(x, format="rgb", quality=quality), channels=3
        )
        x = np.array(x)

        return _get_image_from_arr(x)


class SLQDefense(object):
    doctxt = """
This applies Stochastic Local Quantization with JPEG qualities 20, 40, 60, 80 as defined in [1].

References
[1] Nilaksh Das, Madhuri Shanbhogue, Shang-Tse Chen, Fred Hohman, Siwei Li, Li Chen, Michael E. Kounavis, and Duen Horng Chau. 2018. SHIELD: Fast, Practical Defense and Vaccination for Deep Learning using JPEG Compression. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '18). ACM, New York, NY, USA, 196-204. https://arxiv.org/abs/1802.06816
  """
    option_doctxts = {}

    @staticmethod
    def apply(image):
        image = np.array(image, dtype=np.uint8)
        qualities = (20, 40, 60, 80)
        patch_size = 8

        num_qualities = len(qualities)
        one = tf.constant(1, name="one")
        zero = tf.constant(0, name="zero")

        x = tf.constant(image)
        x_shape = tf.shape(x)
        n, m = x_shape[0], x_shape[1]

        patch_n = (n / patch_size) + tf.cond(
            n % patch_size > 0, lambda: one, lambda: zero
        )
        patch_m = (m / patch_size) + tf.cond(
            n % patch_size > 0, lambda: one, lambda: zero
        )

        R = tf.tile(tf.reshape(tf.range(n), (n, 1)), [1, m])
        C = tf.reshape(tf.tile(tf.range(m), [n]), (n, m))
        Z = tf.compat.v1.image.resize_nearest_neighbor(
            [
                tf.random.uniform(
                    (patch_n, patch_m, 3), 0, num_qualities, dtype=tf.int32
                )
            ],
            (patch_n * patch_size, patch_m * patch_size),
            name="random_layer_indices",
        )[0, :, :, 0][:n, :m]
        indices = tf.transpose(
            tf.stack([Z, R, C]), perm=[1, 2, 0], name="random_layer_indices"
        )

        x_compressed_stack = tf.stack(
            map(
                lambda q: tf.image.decode_jpeg(
                    tf.image.encode_jpeg(x, format="rgb", quality=q), channels=3
                ),
                qualities,
            ),
            name="compressed_images",
        )

        x_slq = tf.gather_nd(x_compressed_stack, indices, name="final_image")
        x_slq = np.array(x_slq)

        return _get_image_from_arr(x_slq)


class MedianFilterDefense(object):
    doctxt = "This applies a median filter of a given window size to the image."
    option_doctxts = {
        "size": (
            "The window size to be used while appying the median filter. "
            "Larger the number, more pixels in the neighborhood will be used "
            "for calculating the median."
        )
    }

    @staticmethod
    def apply(image, size=3):
        image = np.array(image, dtype=np.uint8)
        image = median_filter(image, size=size)
        return _get_image_from_arr(image)


class TVBregmanDefense(object):
    doctxt = (
        "This applies the split-Bregman optimization for the "
        "Total Variational Denoising technique for removing noise from images"
    )
    option_doctxts = {
        "weight": (
            "This is the denoising weight. The smaller the weight, "
            "the more denoising (at the expense of less similarity to the input)."
        )
    }

    @staticmethod
    def apply(image, weight=30.0):
        image = np.array(image, dtype=np.uint8)
        denoised = denoise_tv_bregman(image, weight=weight) * 255.0
        return _get_image_from_arr(denoised)


DEFENSE_MAP = OrderedDict()
DEFENSE_MAP["JPEG"] = JPEGDefense
DEFENSE_MAP["SLQ"] = SLQDefense
DEFENSE_MAP["MedianFilter"] = MedianFilterDefense
DEFENSE_MAP["TV-Bregman"] = TVBregmanDefense
