import tifffile as tf
import numpy as np
import os
import time
import skimage.filters as filters
import skimage
import time
import argparse
from tqdm import tqdm


def get_in_filenames(path):
    files = os.listdir(path)
    files = np.sort([path + i for i in files if i.endswith('.tif') or i.endswith('.tiff')])
    return files

def get_mean_image(path):
    files = get_in_filenames(path)
    running_sum = np.zeros(tf.imread(files[0]).shape)
    print("reading files and generating mean across Z-slices...")
    for i in tqdm(range(len(files))):
        x = tf.imread(files[i])
        running_sum += x
    mean_z = running_sum / float(len(files))
    return mean_z

def get_bias_image(mean, sigma=200):
    print('calculating bias...')
    blurred_mean = filters.gaussian(mean, sigma=sigma)
    bias_z_slice = blurred_mean/mean
    bias_z_slice[ np.isnan(bias_z_slice) ] = 1.0
    return bias_z_slice

def correct_image(img, bias):
    corrected_img = img * bias
    corrected_img[ np.isnan(corrected_img) ] = 1.0
    return corrected_img

def get_out_filenames(files, out_path):
    files = np.sort([os.path.basename(i) for i in files if i.endswith('.tif') or i.endswith('.tiff')])
    out_files = [out_path + i[:i.find('.tif')] + '_corrected.tif' for i in files]
    return np.sort(out_files)


def correct_and_save_images(in_path, out_path, sigma):
    mean = get_mean_image(in_path)
    tf.imsave(out_path + '/mean_image.tif', data=mean.astype('float32'), compress=6)
    bias = get_bias_image(mean, sigma)
    tf.imsave(out_path + '/bias_image.tif', data=bias.astype('float32'), compress=6)
    files = get_in_filenames(in_path)
    out_files = get_out_filenames(files, out_path)
    assert(len(files) == len(out_files))
    print('correcting images...')
    for i in tqdm(range(len(files))):
        img = tf.imread(files[i])
        corrected_img = correct_image(img, bias)
        tf.imsave(out_files[i], data=corrected_img.astype('uint16'), compress=6) 

def main():
    parser = argparse.ArgumentParser(description='Correct gridding artifact in COLM images. Need input directory and output directory')
    parser.add_argument('--in_path', help='Full path to input directory', type=str)
    parser.add_argument('--out_path', help='Full path to output directory', type=str)
    parser.add_argument('--sigma', help='OPTIONAL: intensity of grid correction. higher sigma means more correction, lower sigma means less correction.', default=200, type=float)

    args = parser.parse_args()

    correct_and_save_images(args.in_path, args.out_path, args.sigma)


if __name__ == "__main__":
    main()
