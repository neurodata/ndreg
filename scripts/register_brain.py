# register_brain.py
import numpy as np
import requests
import tifffile as tf
import sys
import os
import ndreg
from ndreg import preprocessor, util
import SimpleITK as sitk
import numpy as np
from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *
import skimage
import argparse
import time
#from ndpull import ndpull
import configparser
from configparser import ConfigParser

dimension = 3
vectorComponentType = sitk.sitkFloat32
vectorType = sitk.sitkVectorFloat32
affine = sitk.AffineTransform(dimension)
identityAffine = list(affine.GetParameters())
identityDirection = identityAffine[0:9]
zeroOrigin = [0] * dimension
zeroIndex = [0] * dimension

ndToSitkDataTypes = {'uint8': sitk.sitkUInt8,
                     'uint16': sitk.sitkUInt16,
                     'uint32': sitk.sitkUInt32,
                     'float32': sitk.sitkFloat32,
                     'uint64': sitk.sitkUInt64}


sitkToNpDataTypes = {sitk.sitkUInt8: np.uint8,
                     sitk.sitkUInt16: np.uint16,
                     sitk.sitkUInt32: np.uint32,
                     sitk.sitkInt8: np.int8,
                     sitk.sitkInt16: np.int16,
                     sitk.sitkInt32: np.int32,
                     sitk.sitkFloat32: np.float32,
                     sitk.sitkFloat64: np.float64,
                     }

# Boss Stuff:


def setup_experiment_boss(remote, collection, experiment):
    """
    Get experiment and coordinate frame information from the boss.
    """
    exp_setup = ExperimentResource(experiment, collection)
    try:
        exp_actual = remote.get_project(exp_setup)
        coord_setup = CoordinateFrameResource(exp_actual.coord_frame)
        coord_actual = remote.get_project(coord_setup)
        return (exp_setup, coord_actual)
    except HTTPError as e:
        print(e.message)


def setup_channel_boss(
        remote,
        collection,
        experiment,
        channel,
        channel_type='image',
        datatype='uint16'):
    (exp_setup, coord_actual) = setup_experiment_boss(
        remote, collection, experiment)

    chan_setup = ChannelResource(
        channel,
        collection,
        experiment,
        channel_type,
        datatype=datatype)
    try:
        chan_actual = remote.get_project(chan_setup)
        return (exp_setup, coord_actual, chan_actual)
    except HTTPError as e:
        print(e.message)


# Note: The following functions assume an anisotropic dataset. This is generally a bad assumption. These
# functions are stopgaps until proper coordinate frame at resulution
# support exists in intern.
def get_xyz_extents(rmt, ch_rsc, res=0, iso=True):
    boss_url = 'https://api.boss.neurodata.io/v1/'
    ds = boss_url + \
        '/downsample/{}?iso={}'.format(ch_rsc.get_cutout_route(), iso)
    headers = {'Authorization': 'Token ' + rmt.token_project}
    r_ds = requests.get(ds, headers=headers)
    response = r_ds.json()
    x_range = [0, response['extent']['{}'.format(res)][0]]
    y_range = [0, response['extent']['{}'.format(res)][1]]
    z_range = [0, response['extent']['{}'.format(res)][2]]
    spacing = response['voxel_size']['{}'.format(res)]
    return (x_range, y_range, z_range, spacing)

def get_offset_boss(coord_frame, res=0, isotropic=False):
    return [
        int(coord_frame.x_start / (2.**res)),
        int(coord_frame.y_start / (2.**res)),
        int(coord_frame.z_start / (2.**res)) if isotropic else coord_frame.z_start]


def get_image_size_boss(coord_frame, res=0, isotropic=False):
    return [
        int(coord_frame.x_stop / (2.**res)),
        int(coord_frame.y_stop / (2.**res)),
        int(coord_frame.z_stop / (2.**res)) if isotropic else coord_frame.z_stop]


def imgDownload_boss(
        remote,
        channel_resource,
        coordinate_frame_resource,
        resolution=0,
        size=[],
        start=[],
        isotropic=False):
    """
    Download image with given token from given server at given resolution.
    If channel isn't specified the first channel is downloaded.
    """
    # TODO: Fix size and start parameters

    voxel_unit = coordinate_frame_resource.voxel_unit
    voxel_units = ('nanometers', 'micrometers', 'millimeters', 'centimeters')
    factor_divide = (1e-6, 1e-3, 1, 10)
    fact_div = factor_divide[voxel_units.index(voxel_unit)]

    spacingBoss = [
        coordinate_frame_resource.x_voxel_size,
        coordinate_frame_resource.y_voxel_size,
        coordinate_frame_resource.z_voxel_size]
    spacing = [x * fact_div for x in spacingBoss]  # Convert spacing to mm
    if isotropic:
        spacing = [x * 2**resolution for x in spacing]
    else:
        spacing[0] = spacing[0] * 2**resolution
        spacing[1] = spacing[1] * 2**resolution
        # z spacing unchanged since not isotropic

    if size == []:
        size = get_image_size_boss(
            coordinate_frame_resource, resolution, isotropic)
    if start == []:
        start = get_offset_boss(
            coordinate_frame_resource, resolution, isotropic)
#    if isotropic:
#        x_range, y_range, z_range, spacing = get_xyz_extents(
#            remote, channel_resource, res=resolution, iso=isotropic)

    # size[2] = 200
    # dataType = metadata['channels'][channel]['datatype']
    dataType = channel_resource.datatype

    # Download all image data from specified channel
    array = remote.get_cutout(
        channel_resource, resolution, [
            start[0], size[0]], [
            start[1], size[1]], [
                start[2], size[2]])

    # Cast downloaded image to server's data type
    # convert numpy array to sitk image
    img = sitk.Cast(sitk.GetImageFromArray(array), ndToSitkDataTypes[dataType])

    # Reverse axes order
    # img = sitk.PermuteAxesImageFilter().Execute(img,range(dimension-1,-1,-1))
    img.SetDirection(identityDirection)
    img.SetSpacing(spacing)

    # Convert to 2D if only one slice
    img = util.imgCollapseDimension(img)

    return img


def get_offset_boss(coord_frame, res=0, isotropic=False):
    return [
        int(coord_frame.x_start / (2.**res)),
        int(coord_frame.y_start / (2.**res)),
        int(coord_frame.z_start / (2.**res)) if isotropic else coord_frame.z_start]

def create_channel_resource(rmt, chan_name, coll_name, exp_name, type='image', 
                            base_resolution=0, sources=[], datatype='uint16', new_channel=True):
    channel_resource = ChannelResource(chan_name, coll_name, exp_name, type=type,
                                       base_resolution=base_resolution, sources=sources, datatype=datatype)
    if new_channel: 
        new_rsc = rmt.create_project(channel_resource)
        return new_rsc

    return channel_resource

def upload_to_boss(rmt, data, channel_resource, resolution=0):
    Z_LOC = 0
    size = data.shape
    for i in range(0, data.shape[Z_LOC], 16):
        last_z = i+16
        if last_z > data.shape[Z_LOC]:
            last_z = data.shape[Z_LOC]
        print(resolution, [0, size[2]], [0, size[1]], [i, last_z])
        rmt.create_cutout(channel_resource, resolution, 
                          [0, size[2]], [0, size[1]], [i, last_z], 
                          np.asarray(data[i:last_z,:,:], order='C'))

def download_ara(rmt, resolution, type='average'):
    if resolution not in [10, 25, 50, 100]:
        print('Please provide a resolution that is among the following: 10, 25, 50, 100')
        return
    REFERENCE_COLLECTION = 'ara_2016'
    REFERENCE_EXPERIMENT = 'sagittal_{}um'.format(resolution)
    REFERENCE_COORDINATE_FRAME = 'ara_2016_{}um'.format(resolution) 
    REFERENCE_CHANNEL = '{}_{}um'.format(type, resolution)

    refImg = download_image(rmt, REFERENCE_COLLECTION, REFERENCE_EXPERIMENT, REFERENCE_CHANNEL, ara_res=resolution)

    return refImg

def download_image(rmt, collection, experiment, channel, res=0, isotropic=True, ara_res=None):
    (exp_resource, coord_resource, channel_resource) = setup_channel_boss(rmt, collection, experiment, channel)
    img = imgDownload_boss(rmt, channel_resource, coord_resource, resolution=res, isotropic=isotropic)
    return img


#def download_image(config_file, collection, experiment, channel, outdir, res=0, isotropic=False, full_extent=True):
#    # conversion factor from mm to um
#    um_to_mm = 1e-3
#
#    # set up ndpull
#    meta = ndpull.BossMeta(collection, experiment, channel, res=res)
#    token, boss_url = ndpull.get_boss_config(config_file)
#    rmt = ndpull.BossRemote(boss_url, token, meta)
#    # get args
#    args = ndpull.collect_input_args(collection, experiment, channel, config_file=config_file, outdir=outdir, res=res, iso=isotropic, full_extent=full_extent)
#    result, rmt = ndpull.validate_args(args)
#    # download slices
#    img = ndpull.download_slices(result, rmt, return_numpy=True, threads=8)
#    img_sitk = sitk.GetImageFromArray(img)
#    coord_frame = rmt.get_coord_frame_metadata()
#    scale_factor = 2.0 ** res
#    vox_sizes = np.array([coord_frame['x_voxel_size'], coord_frame['y_voxel_size'], coord_frame['z_voxel_size']])
#    vox_sizes *= scale_factor * um_to_mm
#    img_sitk.SetSpacing(vox_sizes)
#    return img_sitk
#
def main():
    t_start_overall = time.time()
    parser = argparse.ArgumentParser(description='Register a brain in the BOSS and upload it back in a new experiment.')
    parser.add_argument('--collection', help='Name of collection to upload tif stack to', type=str)
    parser.add_argument('--experiment', help='Name of experiment to upload tif stack to', type=str)
    parser.add_argument('--channel', help='Name of channel to upload tif stack to. Default is new channel will be created unless otherwise specified. See --new_channel', type=str)
    parser.add_argument('--image_orientation', help='Orientation of brain image. 3-letter orientation of brain. For example can be PIR: Posterior, Inferior, Right.', type=str)
    parser.add_argument('--modality', help='The imaging modality the data were collected with. The options are either "colm" or "lavision"', type=str)
    parser.add_argument('--outdir', help='set the directory in which you want to save the intermediates. default is ./{collection}_{experiment}_{channel}_reg', type=str, default=None)
    parser.add_argument('--res', help='Resolution at which to perform the registration in microns. Default is 50', type=int, default=50)
    parser.add_argument('--config', help='Path to configuration file with Boss API token. Default: ~/.intern/intern.cfg', default=os.path.expanduser('~/.intern/intern.cfg'))
    parser.add_argument('--upload_only', help='Flag to perform the upload only and return.', action='store_true')

    args = parser.parse_args()

    if args.upload_only:

        print("uploading annotations to the BOSS")
        anno_channel = 'atlas_{}umreg'.format(args.res)
        source_channel = args.channel
        ch_rsc_og = create_channel_resource(rmt, args.channel, args.collection, args.experiment, new_channel=False)
        print("loading atlas labels")
        anno_10 = tf.imread('./ara_annotation_10um.tif')
        anno_10 = sitk.GetImageFromArray(anno_10.astype('uint32'))
        anno_10.SetSpacing((0.01, 0.01, 0.01))
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
        size_r = (y_size, z_size, x_size)

        print("applying affine transformation to atlas labels")
        img_affine = ndreg.imgApplyAffine(anno_10, trans, spacing=spacing.tolist(), useNearest=True)

        print(img_affine.GetSize())
        print(img_affine.GetSpacing())

        print("applying displacement field transformation to atlas labels")
        img_lddmm = ndreg.imgApplyField(img_affine, field, spacing=spacing.tolist(), 
                                                size=size_r, useNearest=True)
        print(img_lddmm.GetSize())
        print(img_lddmm.GetSpacing())

        # reorient annotations to match image
        print("reorienting the labels to match the image")
        img_lddmm_r = preprocessor.imgReorient(img_lddmm, 'pir', args.image_orientation)
    #    coll_reg = 'cell_detection'
        ch_rsc_anno = create_channel_resource(rmt, anno_channel, args.collection, args.experiment, sources=source_channel, datatype='uint64', type='annotation')
        print("uploading atlas labels to the BOSS")
        anno = sitk.GetArrayFromImage(img_lddmm_r)
        if anno.dtype != 'uint64':
            anno = anno.astype('uint64')
        upload_to_boss(rmt, anno, ch_rsc_anno)


    # mm to um conversion factor
    mm_to_um = 1000.0

    # download image
    rmt = BossRemote(cfg_file_or_dict=args.config)
    # resolution level from 0-6
    if args.modality.lower() == 'colm': 
        resolution_image = 3
        image_isotropic = False
    else: 
        resolution_image = 2
        image_isotropic = True

    # resolution in microns
    resolution_atlas = args.res
    # ensure outdir is default value if None
    if args.outdir is None:
        outdir = '{}_{}_{}_reg/'.format(args.collection, args.experiment, args.channel)
        util.dir_make(outdir)
    else: outdir = args.outdir

    # downloading image
    print('downloading experiment: {}, channel: {}...'.format(args.experiment, args.channel))
    t1 = time.time()
    img = download_image(rmt, args.collection, args.experiment, args.channel, res=resolution_image, isotropic=image_isotropic)
    print("time to download image at res {} um: {} seconds".format(img.GetSpacing()[0] * mm_to_um, time.time()-t1))

    # download atlas
    print('downloading atlas...')
    t1 = time.time()
    atlas = download_ara(rmt, resolution_atlas, type='average')
    print("time to download atlas at {} um: {} seconds".format(resolution_atlas, time.time()-t1))

    print("preprocessing image")
#    img_p = preprocessor.preprocess_brain(img, atlas.GetSpacing(), args.modality, args.image_orientation)
#    # z-axis param in correcting grid is hardcoded assuming z_axis = 2 (3rd axis given original image is IPL)
##    if args.modality.lower() == 'colm': img_p = preprocessor.remove_grid_artifact(img_p, z_axis=2,)
#    img_p.SetOrigin((0.0,0.0,0.0))
#    print("preprocessing done!")
#    print("running registration")
#    assert(img_p.GetOrigin() == atlas.GetOrigin())
#    assert(img_p.GetDirection() == atlas.GetDirection())
#    assert(img_p.GetSpacing() == atlas.GetSpacing())
#
#    atlas_registered = ndreg.register_brain(atlas, img_p, outdir=outdir)
#    print("registration done")
#
#    end_time = time.time()
#
#    print("Overall time taken through all steps: {} seconds ({} minutes)".format(end_time - t_start_overall, (end_time - t_start_overall)/60.0))

#    print("uploading annotations to the BOSS")
    anno_channel = 'atlas_{}umreg'.format(args.res)
    source_channel = args.channel
    ch_rsc_og = create_channel_resource(rmt, args.channel, args.collection, args.experiment, new_channel=False)
#    # set up ndpull
#    meta = ndpull.BossMeta(collection, experiment, channel)
#    token, boss_url = ndpull.get_boss_config(config_file)
#    rmt = ndpull.BossRemote(boss_url, token, meta)
#    # end set up ndpull
    print("loading atlas labels")
    anno_10 = tf.imread('./ara_annotation_10um.tif')
    anno_10 = sitk.GetImageFromArray(anno_10.astype('uint32'))
    anno_10.SetSpacing((0.01, 0.01, 0.01))
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
    size_r = (y_size, x_size, z_size)

    print("applying affine transformation to atlas labels")
    img_affine = ndreg.imgApplyAffine(anno_10, trans, spacing=spacing.tolist(), useNearest=True)

    print(img_affine.GetSize())
    print(img_affine.GetSpacing())

    print("applying displacement field transformation to atlas labels")
    img_lddmm = ndreg.imgApplyField(img_affine, field, spacing=spacing.tolist(), 
                                            size=size_r, useNearest=True)
    print(img_lddmm.GetSize())
    print(img_lddmm.GetSpacing())

    # reorient annotations to match image
    print("reorienting the labels to match the image")
    img_lddmm_r = preprocessor.imgReorient(img_lddmm, 'pir', args.image_orientation)
#    coll_reg = 'cell_detection'
    ch_rsc_anno = create_channel_resource(rmt, anno_channel, args.collection, args.experiment, sources=source_channel, datatype='uint64', type='annotation')
    print("uploading atlas labels to the BOSS")
    anno = sitk.GetArrayFromImage(img_lddmm_r)
    if anno.dtype != 'uint64':
        anno = anno.astype('uint64')
    upload_to_boss(rmt, anno, ch_rsc_anno)


    

if __name__ == "__main__":
    main()
