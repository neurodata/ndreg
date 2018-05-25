import argparse
import os
import ndreg
from ndreg import preprocessor
import numpy as np
import SimpleITK as sitk
from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *
import time
import requests

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

def download_image(rmt, collection, experiment, channel, res=0, isotropic=True, ara_res=None):
    (exp_resource, coord_resource, channel_resource) = setup_channel_boss(rmt, collection, experiment, channel)
    img = imgDownload_boss(rmt, channel_resource, coord_resource, resolution=res, isotropic=isotropic)
    return img


def main():
    t_start_overall = time.time()
    parser = argparse.ArgumentParser(description='Register a brain in the BOSS and upload it back in a new experiment.')
    parser.add_argument('--collection', help='Name of collection to upload tif stack to', type=str)
    parser.add_argument('--experiment', help='Name of experiment to upload tif stack to', type=str)
    parser.add_argument('--channel', help='Name of channel to upload tif stack to. Default is new channel will be created unless otherwise specified. See --new_channel', type=str)
    parser.add_argument('--config', help='Path to configuration file with Boss API token. Default: ~/.intern/intern.cfg', default=os.path.expanduser('~/.intern/intern.cfg'))

    args = parser.parse_args()

    # mm to um conversion factor
    mm_to_um = 1000.0

    # download image
    rmt = BossRemote(cfg_file_or_dict=args.config)

    # resolution level from 0-6
    resolution_image = 0
    image_isotropic = True

    # downloading image
    print('downloading experiment: {}, channel: {}...'.format(args.experiment, args.channel))
    t1 = time.time()
    img = download_image(rmt, args.collection, args.experiment, args.channel, res=resolution_image, isotropic=image_isotropic)
    print("time to download image at res {} um: {} seconds".format(img.GetSpacing()[0] * mm_to_um, time.time()-t1))
    
    print("correction bias in image...")
    t1 = time.time()
    scale = 0.1 # scale at which to perform bias correction
    img_bc = preprocessor.correct_bias_field(img, scale=scale)
    print("time to correct bias in image at res {} um: {} seconds".format(img.GetSpacing()[0] * mm_to_um * (1.0/scale), time.time()-t1))



if __name__ == "__main__":
    main()
