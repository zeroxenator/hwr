import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import numpy as np
import os
from sklearn.utils import resample
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
            # test_flag = False
            ind_src_dir = os.path.join(source_dir, dirs)
            ind_trg_dir = os.path.join(target_dir, dirs)
            os.makedirs(ind_trg_dir)
            os.chdir(ind_src_dir)
            dir = os.getcwd()

            file_index = 0
            for (file) in os.listdir(dir):
                final_path = os.path.join(dir, file)
                image = cv2.imread(final_path, cv2.IMREAD_GRAYSCALE)
                # binarize the images
                _,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
                image = 255 - image
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

    # seq1 = iaa.Sequential([
    #     sometimes(iaa.GaussianBlur(sigma=(0, 3.0)))
    # ])
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

    # images_aug1 = seq1.augment_images(image)
    images_aug2 = seq2.augment_images(image)
    images_aug3 = seq3.augment_images(image)
    images_aug4 = seq4.augment_images(image)
    images_aug5 = seq5.augment_images(image)
    # images_aug6 = seq6.augment_images(inv_image)

    images_aug.append(image)
    # images_aug.append(images_aug1)
    images_aug.append(images_aug2)
    images_aug.append(images_aug3)
    images_aug.append(images_aug4)
    images_aug.append(images_aug5)
    # images_aug.append(inverter(images_aug6))

    return images_aug


def up_sample(target_number=1000):
    base_path = 'monkbrill_aug'
    target_folder = 'monkbrill_aug'
    root_dir = os.getcwd()
    source_dir = os.path.join(root_dir, base_path)
    target_dir = os.path.join(root_dir, target_folder)

    os.chdir(source_dir)
    # all labels
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    test_flag = True
    print(all_subdirs)
    for dirs in all_subdirs:
        if test_flag:
            # test_flag = False # COMMENT THIS LINE TO RUN ALL FOLDERS
            ind_src_dir = os.path.join(source_dir, dirs)
            ind_trg_dir = os.path.join(target_dir, dirs)
            # os.makedirs(ind_trg_dir)
            os.chdir(ind_src_dir)
            dir = os.getcwd()

            file_index = 0
            file_list = os.listdir(dir)
            length = len(file_list)
            extra_sample = target_number - length
            if extra_sample > 0:
                new_file_list = resample(file_list, n_samples=extra_sample, random_state=0)
                for file in new_file_list:
                    final_path = os.path.join(dir, file)
                    image = cv2.imread(final_path, cv2.IMREAD_GRAYSCALE)

                    save_path = os.path.join(ind_trg_dir, str(file_index) + '.jpg')
                    cv2.imwrite(save_path, image)
                    file_index += 1

#load_batch()
up_sample()