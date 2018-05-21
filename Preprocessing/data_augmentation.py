import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_batch():
    base_path = 'monkbrill_jpg'
    target_folder = 'monkbrill_aug'
    root_dir = os.getcwd()
    source_dir = os.path.join(root_dir, base_path)
    target_dir = os.path.join(root_dir, target_folder)

    os.chdir(source_dir)
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    test_flag = True

    for dirs in all_subdirs:
        if test_flag:
            test_flag = False
            ind_src_dir = os.path.join(source_dir, dirs)
            ind_trg_dir = os.path.join(target_dir, dirs)
            os.makedirs(ind_trg_dir)
            os.chdir(ind_src_dir)
            dir = os.getcwd()

            file_index = 0
            for (file) in os.listdir(dir):
                final_path = os.path.join(dir, file)
                image = cv2.imread(final_path, cv2.IMREAD_GRAYSCALE)
                auged_images = image_aug(np.array([image]))

                # loop images to save
                for i in range(len(auged_images)):
                    save_path = os.path.join(ind_trg_dir, str(file_index) + '_' + str(i))
                    cv2.imwrite(save_path + '.jpg', auged_images[i][0])
                file_index += 1


def image_aug(image):
    images_aug = []
    inverter = np.vectorize(lambda img: 255 - img)
    inv_image = inverter(image.copy())

# rotate certain degree
# shift (translate)
# scale
# noise (additive Gaussian noise)
# blur (gaussian, average, median)
# dropout (coarse p=0.2)
#
# shear
# simple noise alpha with edge detect (1.0)
    sometimes = lambda aug: iaa.Sometimes(0.8, aug)

    seq1 = iaa.Sequential([
        sometimes(iaa.GaussianBlur(sigma=(0, 3.0)))
    ])
    seq2 = iaa.Sequential([
        sometimes(iaa.Affine(rotate=(-30, 30)))
    ])
    seq3 = iaa.Sequential([
        sometimes(iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}))
    ])
    seq4 = iaa.Sequential([
        sometimes(iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}))
    ])
    seq5 = iaa.Sequential([
        sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5))
    ])
    seq6 = iaa.Sequential([
        sometimes(iaa.CoarseDropout((0.04, 0.05), size_percent=(0.02, 0.05), per_channel=0.1))
    ])

    images_aug1 = seq1.augment_images(image)
    images_aug2 = seq2.augment_images(image)
    images_aug3 = seq3.augment_images(image)
    images_aug4 = seq4.augment_images(image)
    images_aug5 = seq5.augment_images(image)
    images_aug6 = seq6.augment_images(inv_image)

    images_aug.append(images_aug1)
    images_aug.append(images_aug2)
    images_aug.append(images_aug3)
    images_aug.append(images_aug4)
    images_aug.append(images_aug5)
    images_aug.append(inverter(images_aug6))

    return images_aug


load_batch()

# #Apply heavy augmentations to images (used to create the image at the very top of this readme):
#
# # random example images
# images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
#
# # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
# sometimes = lambda aug: iaa.Sometimes(0.5, aug)
#
# # Define our sequence of augmentation steps that will be applied to every image
# # All augmenters with per_channel=0.5 will sample one value _per image_
# # in 50% of all cases. In all other cases they will sample new values
# # _per channel_.
# seq = iaa.Sequential(
#     [
#         # apply the following augmenters to most images
#         iaa.Fliplr(0.5), # horizontally flip 50% of all images
#         iaa.Flipud(0.2), # vertically flip 20% of all images
#         # crop images by -5% to 10% of their height/width
#         sometimes(iaa.CropAndPad(
#             percent=(-0.05, 0.1),
#             pad_mode=ia.ALL,
#             pad_cval=(0, 255)
#         )),
#         sometimes(iaa.Affine(
#             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
#             translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
#             rotate=(-45, 45), # rotate by -45 to +45 degrees
#             shear=(-16, 16), # shear by -16 to +16 degrees
#             order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
#             cval=(0, 255), # if mode is constant, use a cval between 0 and 255
#             mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
#         )),
#         # execute 0 to 5 of the following (less important) augmenters per image
#         # don't execute all of them, as that would often be way too strong
#         iaa.SomeOf((0, 5),
#             [
#                 sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
#                 iaa.OneOf([
#                     iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
#                     iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
#                     iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
#                 ]),
#                 iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
#                 iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
#                 # search either for all edges or for directed edges,
#                 # blend the result with the original image using a blobby mask
#                 iaa.SimplexNoiseAlpha(iaa.OneOf([
#                     iaa.EdgeDetect(alpha=(0.5, 1.0)),
#                     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
#                 ])),
#                 iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
#                 iaa.OneOf([
#                     iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
#                     iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
#                 ]),
#                 iaa.Invert(0.05, per_channel=True), # invert color channels
#                 iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
#                 iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
#                 # either change the brightness of the whole image (sometimes
#                 # per channel) or change the brightness of subareas
#                 iaa.OneOf([
#                     iaa.Multiply((0.5, 1.5), per_channel=0.5),
#                     iaa.FrequencyNoiseAlpha(
#                         exponent=(-4, 0),
#                         first=iaa.Multiply((0.5, 1.5), per_channel=True),
#                         second=iaa.ContrastNormalization((0.5, 2.0))
#                     )
#                 ]),
#                 iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
#                 iaa.Grayscale(alpha=(0.0, 1.0)),
#                 sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
#                 sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
#                 sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
#             ],
#             random_order=True
#         )
#     ],
#     random_order=True
# )
#
# images_aug = seq.augment_images(images)
#
#
# images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
# seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.GaussianBlur((0, 3.0))])
#
# # show an image with 8*8 augmented versions of image 0
# seq.show_grid(images[0], cols=8, rows=8)
#
# # Show an image with 8*8 augmented versions of image 0 and 8*8 augmented
# # versions of image 1. The identical augmentations will be applied to
# # image 0 and 1.
# seq.show_grid([images[0], images[1]], cols=8, rows=8)