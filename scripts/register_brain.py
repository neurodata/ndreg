# register_brain.py
import numpy as np
import tifffile as tf
import sys
import os
import ndreg
from ndreg import preprocessor, util
import SimpleITK as sitk
from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *
import skimage
import argparse
import time
#from ndpull import ndpull
import configparser
from configparser import ConfigParser

from boss_util import *

def reorient_size(size,in_orient,out_orient):
    o2d = {"r": "r", "l": "r",
                        "s": "s", "i": "s", "a": "a", "p": "a"}
    o1 = [o2d[i.lower()] for i in in_orient]
    o2 = [o2d[i.lower()] for i in out_orient]
    order = np.array([o1.index(i) for i in o2])
    return np.array(size)[order]

def register_data(rmt, config):
    t_start_overall = time.time()


    # mm to um conversion factor
    mm_to_um = 1000.0
    res = 100
    outdir = None

    # download image
    params = config['registration']
    shared_params = config['shared']
    # resolution level from 0-6
    if params['modality'].lower() == 'colm': 
        resolution_image = 3
        image_isotropic = False
    else: 
        resolution_image = 2
        image_isotropic = True

    # resolution in microns
    resolution_atlas = res
    # ensure outdir is default value if None
    if outdir is None:
        outdir = './{}_{}_{}_reg/'.format(shared_params['collection'], shared_params['experiment'], params['channel'])
        util.dir_make(outdir)
    else: outdir = outdir

    # downloading image
    print('downloading experiment: {}, channel: {}...'.format(shared_params['experiment'], params['channel']))
    t1 = time.time()
    img = download_image(rmt, shared_params['collection'], shared_params['experiment'], params['channel'], res=resolution_image, isotropic=image_isotropic)
    print("time to download image at res {} um: {} seconds".format(img.GetSpacing()[0] * mm_to_um, time.time()-t1))

    # download atlas
    print('downloading atlas...')
    t1 = time.time()
    atlas = download_ara(rmt, resolution_atlas, type='average')
    print("time to download atlas at {} um: {} seconds".format(resolution_atlas, time.time()-t1))

    print("preprocessing image")
    img_p = preprocessor.preprocess_brain(img, atlas.GetSpacing(), params['modality'], params['orientation'])
    # z-axis param in correcting grid is hardcoded assuming z_axis = 2 (3rd axis given original image is IPL)
#    if modality.lower() == 'colm': img_p = preprocessor.remove_grid_artifact(img_p, z_axis=2,)
    img_p.SetOrigin((0.0,0.0,0.0))
    print("preprocessing done!")
    print("running registration")
    assert(img_p.GetOrigin() == atlas.GetOrigin())
    assert(img_p.GetDirection() == atlas.GetDirection())
    assert(img_p.GetSpacing() == atlas.GetSpacing())

    atlas_registered = ndreg.register_brain(atlas, img_p, outdir=outdir)
    print("registration done")

    end_time = time.time()

    print("Overall time taken through all steps: {} seconds ({} minutes)".format(end_time - t_start_overall, (end_time - t_start_overall)/60.0))

    print("uploading annotations to the BOSS")
    anno_channel = 'atlas_{}umreg'.format(res)
    source_channel = params['channel']
    ch_rsc_og = create_channel_resource(rmt, params['channel'], shared_params['collection'], shared_params['experiment'], new_channel=False)

    print("loading atlas labels")
    anno_10 = tf.imread('/run/scripts/ara_annotation_10um.tif')
    anno_10 = sitk.GetImageFromArray(anno_10.astype('uint32'))
    anno_10.SetSpacing((0.01, 0.01, 0.01))
    atlas_orientation = 'pir'
    trans = sitk.ReadTransform('{}/atlas_to_observed_affine.txt'.format(outdir))
    field = util.imgRead('{}/lddmm/field.vtk'.format(outdir))
    meta = get_xyz_extents(rmt, ch_rsc_og)
    spacing = np.array(meta[-1])/mm_to_um
    x_size = meta[0][1]
    y_size = meta[1][1]
    z_size = meta[2][1]
    size = (x_size, y_size, z_size)
    # need to reorient size to match atlas
    # i am hard coding the size assuming
    # the image is oriented LPS
#    size_r = (y_size, z_size, x_size)
    # this size is hardocoded assuming
    # input image is IPL (atlas is PIR)
    size_r = reorient_size(size, params['orientation'],atlas_orientation)

    print("applying affine transformation to atlas labels")
    img_affine = ndreg.imgApplyAffine(anno_10, trans, spacing=spacing.tolist(), useNearest=True)

    print("applying displacement field transformation to atlas labels")
    img_lddmm = ndreg.imgApplyField(img_affine, field, spacing=spacing.tolist(), 
                                            size=size_r.tolist(), useNearest=True)
    # reorient annotations to match image
    print("reorienting the labels to match the image")
    img_lddmm_r = preprocessor.imgReorient(img_lddmm, atlas_orientation, params['orientation'])
#    coll_reg = 'cell_detection'
    ch_rsc_anno = create_channel_resource(rmt, anno_channel, shared_params['collection'], shared_params['experiment'], sources=source_channel, datatype='uint64', type='annotation')
    print("uploading atlas labels to the BOSS")
    anno = sitk.GetArrayFromImage(img_lddmm_r)
    if anno.dtype != 'uint64':
        anno = anno.astype('uint64')
    upload_to_boss(rmt, anno, ch_rsc_anno)
    return anno_channel

