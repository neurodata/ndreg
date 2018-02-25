#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import SimpleITK as sitk
import os
import math
import sys
import subprocess
import tempfile
import shutil
import requests
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from intern.remote.boss import BossRemote
from intern.resource.boss.resource import ChannelResource, CoordinateFrameResource, ExperimentResource
import requests
from requests import HTTPError

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


ndregDirPath = os.path.dirname(os.path.realpath(__file__)) + "/"

def _isIterable(variable):
    """
    Returns True if variable is a list, tuple or any other iterable object
    """
    return hasattr(variable, '__iter__')


def _isNumber(variable):
    """
    Returns True if varible is is a number
    """
    try:
        float(variable)
    except TypeError:
        return False
    return True

def run(command, checkReturnValue=True, verbose=False):
    """
    Runs a shell command and returns the output.
    """
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1)
    outText = ""

    for line in iter(process.stdout.readline, ''):
        if verbose:
            sys.stdout.write(line)
        outText += line
    # process.wait()
    process.communicate()[0]
    returnValue = process.returncode
    if checkReturnValue and (returnValue != 0):
        raise Exception(outText)

    return (returnValue, outText)


def txtWrite(text, path, mode="w"):
    """
    Conveinence function to write text to a file at specified path
    """
    dirMake(os.path.dirname(path))
    textFile = open(path, mode)
    print(text, file=textFile)
    textFile.close()


def txtRead(path):
    """
    Conveinence function to read text from file at specified path
    """
    textFile = open(path, "r")
    text = textFile.read()
    textFile.close()
    return text


def txtReadList(path):
    return map(float, txtRead(path).split())


def txtWriteList(parameterList, path):
    txtWrite(" ".join(map(str, parameterList)), path)


def dirMake(dirPath):
    """
    Convenience function to create a directory at the given path
    """
    if dirPath != "":
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        return os.path.normpath(dirPath) + "/"
    else:
        return dirPath


def imgHM(inImg, refImg, numMatchPoints=64, numBins=256):
    """
    Histogram matches input image to reference image and writes result to output image
    """
    inImg = sitk.Cast(inImg, refImg.GetPixelID())
    return sitk.HistogramMatchingImageFilter().Execute(
        inImg, refImg, numBins, numMatchPoints, False)


def imgRead(path):
    """
    Alias for sitk.ReadImage
    """

    inImg = sitk.ReadImage(path)
    inImg = imgCollaspeDimension(inImg)
    # if(inImg.GetDimension() == 2): inImg =
    # sitk.JoinSeriesImageFilter().Execute(inImg)

    inDimension = inImg.GetDimension()
    inImg.SetDirection(sitk.AffineTransform(inDimension).GetMatrix())
    inImg.SetOrigin([0] * inDimension)

    return inImg

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
    img = imgCollaspeDimension(img)

    return img


def get_offset_boss(coord_frame, res=0, isotropic=False):
    return [
        int(coord_frame.x_start / (2.**res)),
        int(coord_frame.y_start / (2.**res)),
        int(coord_frame.z_start / (2.**res)) if isotropic else coord_frame.z_start]

def imgUpload_boss(
        remote,
        img,
        channel_resource,
        coord_frame,
        resolution=0,
        start=[
            0,
            0,
            0],
        propagate=False,
        isotropic=False):
    if(img.GetDimension() == 2):
        img = sitk.JoinSeriesImageFilter().Execute(img)

    data = sitk.GetArrayFromImage(img)  # data is C-ordered (z y x)

    offset = get_offset_boss(coord_frame, resolution, isotropic)

    start = [x + y for x, y in zip(start, offset)]

    st_x = start[0]
    st_y = start[1]
    st_z = start[2]
    sp_x = st_x + np.shape(data)[2]
    sp_y = st_y + np.shape(data)[1]
    sp_z = st_z + np.shape(data)[0]

    try:
        remote.create_cutout(channel_resource, resolution,
                             [st_x, sp_x], [st_y, sp_y], [st_z, sp_z], data)
        print('Upload success')
    except Exception as e:
        # perhaps reconnect, etc.
        print('Exception occurred: {}'.format(e))
        raise(e)

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
    REFERENCE_CHANNEL = '{}_{}um'.format(type, resolution)

    refImg = download_image(rmt, REFERENCE_COLLECTION, REFERENCE_EXPERIMENT, REFERENCE_CHANNEL, ara_res=resolution)

    return refImg

def download_image(rmt, collection, experiment, channel, res=0, isotropic=True, ara_res=None):
    (_, coord_resource, channel_resource) = setup_channel_boss(rmt, collection, experiment, channel)
    img = imgDownload_boss(rmt, channel_resource, coord_resource, resolution=res, isotropic=isotropic)
    return img

def imgCopy(img):
    """
    Returns a copy of the input image
    """
    return sitk.Image(img)


def imgWrite(img, path):
    """
    Write sitk image to path.
    """
    dirMake(os.path.dirname(path))
    sitk.WriteImage(img, path)

    # Reformat files to be compatible with CIS Software
    ext = os.path.splitext(path)[1].lower()
    if ext == ".vtk":
        vtkReformat(path, path)


def vtkReformat(inPath, outPath):
    """
    Reformats vtk file so that it can be read by CIS software.
    """
    # Get size of map
    inFile = open(inPath, "rb")
    lineList = inFile.readlines()
    for line in lineList:
        if line.lower().strip().startswith("dimensions"):
            size = map(int, line.split(" ")[1:dimension + 1])
            break
    inFile.close()

    if dimension == 2:
        size += [0]

    outFile = open(outPath, "wb")
    for (i, line) in enumerate(lineList):
        if i == 1:
            newline = line.lstrip(line.rstrip("\n"))
            line = "lddmm 8 0 0 {0} {0} 0 0 {1} {1} 0 0 {2} {2}".format(
                size[2] - 1, size[1] - 1, size[0] - 1) + newline
        outFile.write(line)


def imgResample(img, spacing, size=[], useNearest=False,
                origin=[], outsideValue=0):
    """
    Resamples image to given spacing and size.
    """
    if len(spacing) != img.GetDimension():
        raise Exception(
            "len(spacing) != " + str(img.GetDimension()))

    # Set Size
    if size == []:
        inSpacing = img.GetSpacing()
        inSize = img.GetSize()
        size = [int(math.ceil(inSize[i] * (inSpacing[i] / spacing[i])))
                for i in range(img.GetDimension())]
    else:
        if len(size) != img.GetDimension():
            raise Exception(
                "len(size) != " + str(img.GetDimension()))

    if origin == []:
        origin = [0] * img.GetDimension()
    else:
        if len(origin) != img.GetDimension():
            raise Exception(
                "len(origin) != " + str(img.GetDimension()))

    # Resample input image
    interpolator = [sitk.sitkBSpline, sitk.sitkNearestNeighbor][useNearest]
    identityTransform = sitk.Transform()
    identityDirection = list(
        sitk.AffineTransform(
            img.GetDimension()).GetMatrix())

    return sitk.Resample(
        img,
        size,
        identityTransform,
        interpolator,
        origin,
        spacing,
        identityDirection,
        outsideValue)


#def imgPad(img, padding=0, useNearest=False):
#    """
#    Pads image by given ammount of padding in units spacing.
#    For example if the input image has a voxel spacing of 0.5 and the padding=2.0 then the image will be padded by 4 voxels.
#    If the padding < 0 then the filter crops the image
#    """
#    if _isNumber(padding):
#        padding = [padding] * img.GetDimension()
#    elif len(padding) != img.GetDimension():
#        raise Exception(
#            "padding must have length {0}.".format(img.GetDimension()))
#
#    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]
#    translationTransform = sitk.TranslationTransform(
#        img.GetDimension(), -np.array(padding))
#    spacing = img.GetSpacing()
#    size = list(img.GetSize())
#    for i in range(img.GetDimension()):
#        if padding[i] > 0:
#            paddingVoxel = int(math.ceil(2 * padding[i] / spacing[i]))
#        else:
#            paddingVoxel = int(math.floor(2 * padding[i] / spacing[i]))
#        size[i] += paddingVoxel
#
#    origin = [0] * img.GetDimension()
#    return sitk.Resample(img, size, translationTransform,
#                         interpolator, origin, spacing)
#
def createTmpRegistration(inMask=None, refMask=None,
                          samplingFraction=1.0, dimension=dimension):
    identityTransform = sitk.Transform(dimension, sitk.sitkIdentity)
    tmpRegistration = sitk.ImageRegistrationMethod()
    tmpRegistration.SetInterpolator(sitk.sitkNearestNeighbor)
    tmpRegistration.SetInitialTransform(identityTransform)
    tmpRegistration.SetOptimizerAsGradientDescent(
        learningRate=1e-14, numberOfIterations=1)
    if samplingFraction != 1.0:
        tmpRegistration.SetMetricSamplingPercentage(samplingFraction)
        tmpRegistration.SetMetricSamplingPercentage(tmpRegistration.RANDOM)

    if(inMask):
        tmpRegistration.SetMetricMovingMask(inMask)
    if(refMask):
        tmpRegistration.SetMetricFixedMask(refMask)

    return tmpRegistration

def imgCollaspeDimension(inImg):
    inSize = inImg.GetSize()

    if inImg.GetDimension() == dimension and inSize[dimension - 1] == 1:
        outSize = list(inSize)
        outSize[dimension - 1] = 0
        outIndex = [0] * dimension
        inImg = sitk.Extract(inImg, outSize, outIndex, 1)

    return inImg

def imgNorm(img):
    """
    Returns the L2-Norm of an image
    """
    if img.GetNumberOfComponentsPerPixel() > 1:
        img = sitk.VectorMagnitude(img)
    stats = sitk.StatisticsImageFilter()
    stats.Execute(img)
    return stats.GetSum()


def imgMI(inImg, refImg, inMask=None, refMask=None,
          numBins=128, samplingFraction=1.0):
    """
    Compute mattes mutual information between input and reference images
    """

    # In SimpleITK the metric can't be accessed directly.
    # Therefore we create a do-nothing registration method which uses an
    # identity transform to get the metric value
    inImg = imgCollaspeDimension(inImg)
    refImg = imgCollaspeDimension(refImg)
    if inMask:
        imgCollaspeDimension(inMask)
    if refMask:
        imgCollaspeDimension(refMask)

    tmpRegistration = createTmpRegistration(
        inMask, refMask, dimension=inImg.GetDimension(), samplingFraction=samplingFraction)
    tmpRegistration.SetMetricAsMattesMutualInformation(numBins)
    tmpRegistration.Execute(
        sitk.Cast(refImg, sitk.sitkFloat32), sitk.Cast(inImg, sitk.sitkFloat32))

    return -tmpRegistration.GetMetricValue()


def imgMSE(inImg, refImg, inMask=None, refMask=None, samplingFraction=1.0):
    """
    Compute mean square error between input and reference images
    """
    inImg = imgCollaspeDimension(inImg)
    refImg = imgCollaspeDimension(refImg)
    if inMask:
        imgCollaspeDimension(inMask)
    if refMask:
        imgCollaspeDimension(refMask)
    tmpRegistration = createTmpRegistration(
        inMask, refMask, dimension=refImg.GetDimension(), samplingFraction=1.0)
    tmpRegistration.SetMetricAsMeanSquares()
    tmpRegistration.Execute(
        sitk.Cast(refImg, sitk.sitkFloat32), sitk.Cast(inImg, sitk.sitkFloat32))

    return tmpRegistration.GetMetricValue()

def imgMask(img, mask):
    """
    Convenience function to apply mask to image
    """
    mask = imgResample(mask, img.GetSpacing(), img.GetSize(), useNearest=True)
    mask = sitk.Cast(mask, img.GetPixelID())
    return sitk.MaskImageFilter().Execute(img, mask)

def imgApplyField(img, field, useNearest=False,
                  size=[], spacing=[], defaultValue=0):
    """
    img \circ field
    """
    field = sitk.Cast(field, sitk.sitkVectorFloat64)

    # Set interpolator
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]

    # Set transform field
    transform = sitk.DisplacementFieldTransform(img.GetDimension())
    transform.SetInterpolator(sitk.sitkLinear)
    transform.SetDisplacementField(field)

    # Set size
    if size == []:
        size = img.GetSize()
    else:
        if len(size) != img.GetDimension():
            raise Exception(
                "size must have length {0}.".format(img.GetDimension()))

    # Set Spacing
    if spacing == []:
        spacing = img.GetSpacing()
    else:
        if len(spacing) != img.GetDimension():
            raise Exception(
                "spacing must have length {0}.".format(img.GetDimension()))

    # Apply displacement transform
    return sitk.Resample(img, size, transform, interpolator, [
                         0] * img.GetDimension(), spacing, img.GetDirection(), defaultValue)


def imgApplyAffine(inImg, affine, useNearest=False, size=[], spacing=[]):
    inDimension = inImg.GetDimension()

    # Set interpolator
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]

    # Set affine parameters
    affineTransform = sitk.AffineTransform(inDimension)
    numParameters = len(affineTransform.GetParameters())
    if (len(affine) != numParameters):
        raise Exception(
            "affine must have length {0}.".format(numParameters))
    affineTransform = sitk.AffineTransform(inDimension)
    affineTransform.SetParameters(affine)

    # Set Spacing
    if spacing == []:
        spacing = inImg.GetSpacing()
    else:
        if len(spacing) != inDimension:
            raise Exception(
                "spacing must have length {0}.".format(inDimension))

    # Set size
    if size == []:
        # Compute size to contain entire output image
        size = sizeOut(inImg, affineTransform, spacing)
    else:
        if len(size) != inDimension:
            raise Exception(
                "size must have length {0}.".format(inDimension))

    # Apply affine transform
    outImg = sitk.Resample(inImg, size, affineTransform,
                           interpolator, zeroOrigin, spacing)

    return outImg

def fieldApplyField(inField, field, size=[], spacing=[]):
    """ outField = inField \circ field """
    inField = sitk.Cast(inField, sitk.sitkVectorFloat64)
    field = sitk.Cast(field, sitk.sitkVectorFloat64)
    inDimension = inField.GetDimension()

    if spacing == []:
        spacing = list(inField.GetSpacing())
    else:
        if len(spacing) != inDimension:
            raise Exception(
                "spacing must have length {0}.".format(inDimension))

    # Set size
    if size == []:
        # Compute size to contain entire output image
        size = list(inField.GetSize())
    else:
        if len(size) != inDimension:
            raise Exception(
                "size must have length {0}.".format(inDimension))

    # Create transform for input field
    inTransform = sitk.DisplacementFieldTransform(inDimension)
    inTransform.SetDisplacementField(inField)
    inTransform.SetInterpolator(sitk.sitkLinear)

    # Create transform for field
    transform = sitk.DisplacementFieldTransform(inDimension)
    transform.SetDisplacementField(field)
    transform.SetInterpolator(sitk.sitkLinear)

    # Combine transforms
    outTransform = sitk.Transform(transform)
    # outTransform.AddTransform(transform)
    outTransform.AddTransform(inTransform)

    # Get output displacement field
    direction = np.eye(inDimension).flatten().tolist()
    origin = [0] * inDimension
    return sitk.TransformToDisplacementFieldFilter().Execute(
        outTransform, vectorType, size, origin, spacing, direction)


def imgReorient(inImg, inOrient, outOrient):
    """
    Reorients image from input orientation inOrient to output orientation outOrient.
    inOrient and outOrient must be orientation strings specifying the orientation of the image.
    For example an orientation string of "las" means that the ...
        x-axis increases from \"l\"eft to right
        y-axis increases from \"a\"nterior to posterior
        z-axis increases from \"s\"uperior to inferior
    Thus using inOrient = "las" and outOrient = "rpi" reorients the input image from left-anterior-superior to right-posterior-inferior orientation.
    """
    if (len(inOrient) != dimension) or not isinstance(inOrient, basestring):
        raise Exception(
            "inOrient must be a string of length {0}.".format(dimension))
    if (len(outOrient) != dimension) or not isinstance(outOrient, basestring):
        raise Exception(
            "outOrient must be a string of length {0}.".format(dimension))
    inOrient = str(inOrient).lower()
    outOrient = str(outOrient).lower()

    inDirection = ""
    outDirection = ""
    orientToDirection = {"r": "r", "l": "r",
                         "s": "s", "i": "s", "a": "a", "p": "a"}
    for i in range(dimension):
        try:
            inDirection += orientToDirection[inOrient[i]]
        except BaseException:
            raise Exception("inOrient \'{0}\' is invalid.".format(inOrient))

        try:
            outDirection += orientToDirection[outOrient[i]]
        except BaseException:
            raise Exception("outOrient \'{0}\' is invalid.".format(outOrient))

    if len(set(inDirection)) != dimension:
        raise Exception(
            "inOrient \'{0}\' is invalid.".format(inOrient))
    if len(set(outDirection)) != dimension:
        raise Exception(
            "outOrient \'{0}\' is invalid.".format(outOrient))

    order = []
    flip = []
    for i in range(dimension):
        j = inDirection.find(outDirection[i])
        order += [j]
        flip += [inOrient[j] != outOrient[i]]

    outImg = sitk.PermuteAxesImageFilter().Execute(inImg, order)
    outImg = sitk.FlipImageFilter().Execute(outImg, flip, False)
    outImg.SetDirection(identityDirection)
    outImg.SetOrigin(zeroOrigin)
    return outImg


def imgChecker(inImg, refImg, useHM=True, pattern=[4] * dimension):
    """
    Checkerboards input image with reference image
    """
    inImg = sitk.Cast(inImg, refImg.GetPixelID())
    inSize = list(inImg.GetSize())
    refSize = list(refImg.GetSize())

    if(inSize != refSize):
        sourceSize = np.array([inSize, refSize]).min(0)
        # Empty image with same size as reference image
        tmpImg = sitk.Image(refSize, refImg.GetPixelID())
        tmpImg.CopyInformation(refImg)
        inImg = sitk.PasteImageFilter().Execute(
            tmpImg, inImg, sourceSize, zeroIndex, zeroIndex)

    if useHM:
        inImg = imgHM(inImg, refImg)

    return sitk.CheckerBoardImageFilter().Execute(inImg, refImg, pattern)

def imgMetamorphosis(inImg, refImg, alpha=0.02, beta=0.05, scale=1.0, iterations=1000, epsilon=None, minEpsilon=None, sigma=1e-4, useNearest=False,
                     useBias=False, useMI=False, verbose=False, debug=False, inMask=None, refMask=None, outDirPath=""):
    """
    Performs Metamorphic LDDMM between input and reference images
    """
    useTempDir = False
    if outDirPath == "":
        useTempDir = True
        outDirPath = tempfile.mkdtemp() + "/"
    else:
        outDirPath = dirMake(outDirPath)

    inPath = outDirPath + "in.img"
    imgWrite(inImg, inPath)
    refPath = outDirPath + "ref.img"
    imgWrite(refImg, refPath)
    outPath = outDirPath + "out.img"

    fieldPath = outDirPath + "field.vtk"
    invFieldPath = outDirPath + "invField.vtk"

    binPath = ndregDirPath + "metamorphosis "
    steps = 5
    command = binPath + " --in {0} --ref {1} --out {2} --alpha {3} --beta {4} --field {5} --invfield {6} --iterations {7} --scale {8} --steps {9} --verbose ".format(
        inPath, refPath, outPath, alpha, beta, fieldPath, invFieldPath, iterations, scale, steps)
    if(not useBias):
        command += " --mu 0"
    if(useMI):
        # command += " --cost 1 --sigma 1e-5 --epsilon 1e-3"
        command += " --cost 1 --sigma {}".format(sigma)
#         command += " --epsilon {0}".format(epsilon)
        if not(epsilon is None):
            command += " --epsilon {0}".format(epsilon)
        else:
            command += " --epsilon 1e-3"
    else:
        command += " --sigma {}".format(sigma)
#         command += " --epsilon {0}".format(epsilon)
        if not(epsilon is None):
            command += " --epsilon {0}".format(epsilon)
        else:
            command += " --epsilon 1e-3"

    if not(minEpsilon is None):
        command += " --epsilonmin {0}".format(minEpsilon)

    if(inMask):
        inMaskPath = outDirPath + "inMask.img"
        imgWrite(inMask, inMaskPath)
        command += " --inmask " + inMaskPath

    if(refMask):
        refMaskPath = outDirPath + "refMask.img"
        imgWrite(refMask, refMaskPath)
        command += " --refmask " + refMaskPath

    if debug:
        command = "/usr/bin/time -v " + command
        print(command)

    # os.system(command)
    (returnValue, logText) = run(command, verbose=verbose)

    logPath = outDirPath + "log.txt"
    txtWrite(logText, logPath)

    field = imgRead(fieldPath)
    invField = imgRead(invFieldPath)

    if useTempDir:
        shutil.rmtree(outDirPath)
    return (field, invField)


def imgMetamorphosisComposite(inImg, refImg, alphaList=0.02, betaList=0.05, scaleList=1.0, iterations=1000, epsilonList=None, minEpsilonList=None, sigma=1e-4,
                              useNearest=False, useBias=False, useMI=False, inMask=None, refMask=None, verbose=True, debug=False, outDirPath=""):
    """
    Performs Metamorphic LDDMM between input and reference images
    """
    useTempDir = False
    if outDirPath == "":
        useTempDir = True
        outDirPath = tempfile.mkdtemp() + "/"
    else:
        outDirPath = dirMake(outDirPath)

    if _isNumber(alphaList):
        alphaList = [float(alphaList)]
    if _isNumber(betaList):
        betaList = [float(betaList)]
    if _isNumber(scaleList):
        scaleList = [float(scaleList)]

    numSteps = max(len(alphaList), len(betaList), len(scaleList))

    if _isNumber(epsilonList):
        epsilonList = [float(epsilonList)] * numSteps
    elif epsilonList is None:
        epsilonList = [None] * numSteps

    if _isNumber(minEpsilonList):
        minEpsilonList = [float(minEpsilonList)] * numSteps
    elif minEpsilonList is None:
        minEpsilonList = [None] * numSteps

    if len(alphaList) != numSteps:
        if len(alphaList) != 1:
            raise Exception(
                "Legth of alphaList must be 1 or same length as betaList or scaleList")
        else:
            alphaList *= numSteps

    if len(betaList) != numSteps:
        if len(betaList) != 1:
            raise Exception(
                "Legth of betaList must be 1 or same length as alphaList or scaleList")
        else:
            betaList *= numSteps

    if len(scaleList) != numSteps:
        if len(scaleList) != 1:
            raise Exception(
                "Legth of scaleList must be 1 or same length as alphaList or betaList")
        else:
            scaleList *= numSteps

    origInImg = inImg
    origInMask = inMask
    for step in range(numSteps):
        alpha = alphaList[step]
        beta = betaList[step]
        scale = scaleList[step]
        epsilon = epsilonList[step]
        minEpsilon = minEpsilonList[step]
        stepDirPath = outDirPath + "step" + str(step) + "/"
        if(verbose):
            print("\nStep {0}: alpha={1}, beta={2}, scale={3}".format(
                step, alpha, beta, scale))

        (field, invField) = imgMetamorphosis(inImg, refImg,
                                             alpha,
                                             beta,
                                             scale,
                                             iterations,
                                             epsilon,
                                             minEpsilon,
                                             sigma,
                                             useNearest,
                                             useBias,
                                             useMI,
                                             verbose,
                                             debug,
                                             inMask=inMask,
                                             refMask=refMask,
                                             outDirPath=stepDirPath)

        if step == 0:
            compositeField = field
            compositeInvField = invField
        else:
            compositeField = fieldApplyField(field, compositeField)
            compositeInvField = fieldApplyField(compositeInvField, invField, size=field.GetSize(
            ), spacing=field.GetSpacing())  # force invField to be same size as field

            if outDirPath != "":
                fieldPath = stepDirPath + "field.vtk"
                invFieldPath = stepDirPath + "invField.vtk"
                imgWrite(compositeInvField, invFieldPath)
                imgWrite(compositeField, fieldPath)

        inImg = imgApplyField(origInImg, compositeField, size=refImg.GetSize())
        if(inMask):
            inMask = imgApplyField(origInMask,
                                   compositeField, size=refImg.GetSize(), useNearest=True)
        # vikram added this
#        if verbose: imgShow(inImg, vmax=imgPercentile(inImg, 0.99))

    # Write final results
    if outDirPath != "":
        imgWrite(compositeField, outDirPath + "field.vtk")
        imgWrite(compositeInvField, outDirPath + "invField.vtk")
        imgWrite(inImg, outDirPath + "out.img")
        imgWrite(imgChecker(inImg, refImg), outDirPath + "checker.img")

    if useTempDir:
        shutil.rmtree(outDirPath)
    return (compositeField, compositeInvField)

def imgShow(img, vmin=None, vmax=None, cmap=None, alpha=None,
            newFig=True, flip=[0, 0, 0], numSlices=3, useNearest=False):
    """
    Displays an image.  Only 2D images are supported for now
    """
    if newFig:
        fig = plt.figure()

    if (vmin is None) or (vmax is None):
        stats = sitk.StatisticsImageFilter()
        stats.Execute(img)
        if vmin is None:
            vmin = stats.GetMinimum()
        if vmax is None:
            vmax = stats.GetMaximum()

    if cmap is None:
        cmap = plt.cm.gray
    if alpha is None:
        alpha = 1.0

    interpolation = ['bilinear', 'none'][useNearest]

    if img.GetDimension() == 2:
        plt.axis('off')
        ax = plt.imshow(sitk.GetArrayFromImage(img), cmap=cmap, vmin=vmin,
                        vmax=vmax, alpha=alpha, interpolation=interpolation)

    elif img.GetDimension() == 3:
        size = img.GetSize()
        for i in range(img.GetDimension()):
            start = size[2 - i] / (numSlices + 1)
            sliceList = np.linspace(start, size[2 - i] - start, numSlices)
            sliceSize = list(size)
            sliceSize[2 - i] = 0

            for (j, slice) in enumerate(sliceList):
                sliceIndex = [0] * img.GetDimension()
                sliceIndex[2 - i] = int(slice)
                sliceImg = sitk.Extract(img, sliceSize, sliceIndex)
                sliceArray = sitk.GetArrayFromImage(sliceImg)
                if flip[i]:
                    sliceArray = np.transpose(sliceArray)

                plt.subplot(numSlices, img.GetDimension(),
                            i + img.GetDimension() * j + 1)
                ax = plt.imshow(sliceArray, cmap=cmap, vmin=vmin,
                                vmax=vmax, alpha=alpha, interpolation=interpolation)
                plt.axis('off')
    else:
        raise Exception("Image dimension must be 2 or 3.")

    if newFig:
        plt.show()


def imgShowResults(inImg, refImg, field, logPath=""):
    numRows = 5
    numCols = 3
    defInImg = imgApplyField(inImg, field, size=refImg.GetSize())
    checker = imgChecker(defInImg, refImg)

    sliceList = []
    for i in range(inImg.GetDimension()):
        step = [5] * dimension
        step[2 - i] = None
        grid = imgGrid(inImg.GetSize(), inImg.GetSpacing(),
                       step=step, field=field)

        sliceList.append(imgSlices(grid, flip=[0, 1, 1])[i])
    fig = plt.figure()
    imgShowResultsRow(inImg, numRows, numCols, 0, title="$I_0$")
    imgShowResultsRow(defInImg, numRows, numCols, 1,
                      title="$I_0 \circ \phi_{10}$")
    imgShowResultsRow(checker, numRows, numCols, 2,
                      title="$I_0$ and $I_1$\n Checker")
    imgShowResultsRow(refImg, numRows, numCols, 3, title="$I_1$")
    imgShowResultsRow(sliceList, numRows, numCols, 4, title="$\phi_{10}$")
    fig.subplots_adjust(hspace=0.05, wspace=0)
    plt.show()


def imgShowResultsRow(img, numRows=1, numCols=3, rowIndex=0, title=""):
    if isinstance(img, list):
        sliceImgList = img
    else:
        sliceImgList = imgSlices(img, flip=[0, 1, 1])

    for (i, sliceImg) in enumerate(sliceImgList):
        ax = plt.subplot(numRows, numCols, rowIndex * numCols + i + 1)
        plt.imshow(sitk.GetArrayFromImage(sliceImg),
                   cmap=plt.cm.gray, aspect='auto')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        if i == 0:
            plt.ylabel(title, rotation=0, labelpad=30)
        # plt.axis('off')


def imgGrid(size, spacing, step=[10, 10, 10], field=None):
    """
    Creates a grid image using with specified size and spacing with distance between lines defined by step.
    If step is None along a dimention no grid lines will be plotted.
    For example step=[5,5,None] will result in a grid image with grid lines every 5 voxels in the x and y directions but no grid lines in the z direction.
    An optinal displacement field can be applied to the grid as well.
    """

    if not(_isIterable(size)):
        raise Exception("size must be a list.")
    if not(_isIterable(spacing)):
        raise Exception("spacing must be a list.")
    if not(_isIterable(step)):
        raise Exception("step must be a list.")
    if len(size) != len(spacing):
        raise Exception("len(size) != len(spacing)")
    if len(size) != len(step):
        raiseException("len(size) != len(step)")

    dimension = len(size)
    offset = [0] * dimension

    for i in range(dimension):
        if step[i] is None:
            step[i] = size[i] + 2
            offset[i] = -1

    gridSource = sitk.GridImageSource()
    gridSource.SetSpacing(spacing)
    gridSource.SetGridOffset(np.array(offset) * np.array(spacing))
    gridSource.SetOrigin([0] * dimension)
    gridSource.SetSize(np.array(size))
    gridSource.SetGridSpacing(np.array(step) * np.array(spacing))
    gridSource.SetScale(255)
    gridSource.SetSigma(1 * np.array(spacing))
    grid = gridSource.Execute()

    if not(field is None):
        grid = sitk.WrapPad(grid, [20] * dimension, [20] * dimension)
        grid = imgApplyField(grid, field, size=size)

    return grid


def imgSlices(img, flip=[0, 0, 0], numSlices=1):
    size = img.GetSize()
    sliceImgList = []
    for i in range(img.GetDimension()):
        start = size[2 - i] / (numSlices + 1)
        sliceList = np.linspace(start, size[2 - i] - start, numSlices)
        sliceSize = list(size)
        sliceSize[2 - i] = 0

        for (j, slice) in enumerate(sliceList):
            sliceIndex = [0] * img.GetDimension()
            sliceIndex[2 - i] = int(slice)
            sliceImg = sitk.Extract(img, sliceSize, sliceIndex)

            if flip[i]:
                sliceImgDirection = sliceImg.GetDirection()
                sliceImg = sitk.PermuteAxesImageFilter().Execute(
                    sliceImg, range(sliceImg.GetDimension() - 1, -1, -1))
                sliceImg.SetDirection(sliceImgDirection)
            sliceImgList.append(sliceImg)

    return sliceImgList


def imgPercentile(img, percentile):
    if percentile < 0.0 or percentile > 1.0:
        raise Exception("Percentile should be between 0.0 and 1.0")

    (values, bins) = np.histogram(sitk.GetArrayFromImage(img), bins=255)
    cumValues = np.cumsum(values).astype(float)
    cumValues = (cumValues - cumValues.min()) / cumValues.ptp()

    index = np.argmax(cumValues > percentile) - 1
    value = bins[index]
    return value


def imgMetamorphosisSlicePlotterRow(
        img, numRows=1, numCols=3, rowIndex=0, title="", vmin=None, vmax=None):
    if isinstance(img, list):
        sliceImgList = img
    else:
        if vmax is None or (vmin is None):
            stats = sitk.StatisticsImageFilter()
            stats.Execute(img)
            if vmin is None:
                vmin = stats.GetMinimum()
            if vmax is None:
                vmax = stats.GetMaximum()
        sliceImgList = imgSlices(img, flip=[0, 1, 1])

    for (i, sliceImg) in enumerate(sliceImgList):
        ax = plt.subplot(numRows, numCols, rowIndex * numCols + i + 1)
        plt.imshow(
            sitk.GetArrayFromImage(sliceImg),
            cmap=plt.cm.gray,
            aspect='auto',
            vmax=vmax,
            vmin=vmin)
        ax.set_yticks([])
        ax.set_xticks([])
        if i == 0:
            plt.ylabel(title, rotation=0, labelpad=40)


def imgMetamorphosisSlicePlotter(inImg, refImg, field):
    numRows = 5
    numCols = 3
    defInImg = imgApplyField(inImg, field, size=refImg.GetSize())
    inImg = imgApplyAffine(
        inImg, [
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], size=refImg.GetSize())
    checker = imgChecker(defInImg, refImg)

    sliceList = []
    for i in range(inImg.GetDimension()):
        step = [20] * dimension
        step[2 - i] = None
        grid = imgGrid(
            inImg.GetSize(),
            inImg.GetSpacing(),
            step=step,
            field=field)

        sliceList.append(imgSlices(grid, flip=[0, 1, 1])[i])

    imgMetamorphosisSlicePlotterRow(
        inImg,
        numRows,
        numCols,
        0,
        title="$I_0$",
        vmax=imgPercentile(
            inImg,
            0.99))
    imgMetamorphosisSlicePlotterRow(
        defInImg,
        numRows,
        numCols,
        1,
        title="$I(1)$",
        vmax=imgPercentile(
            defInImg,
            0.99))
    imgMetamorphosisSlicePlotterRow(
        checker,
        numRows,
        numCols,
        2,
        title="$I(1)$ and $I_1$\n Checker",
        vmax=imgPercentile(
            checker,
            0.99))
    imgMetamorphosisSlicePlotterRow(
        refImg,
        numRows,
        numCols,
        3,
        title="$I_1$",
        vmax=imgPercentile(
            refImg,
            0.99))
    imgMetamorphosisSlicePlotterRow(
        sliceList, numRows, numCols, 4, title="$\phi_{10}$")
    plt.gcf().subplots_adjust(hspace=0.1, wspace=0.025)


def imgMetamorphosisLogPlotter(
        logPathList, labelList=None, useLog=False, useTime=False):
    if not(_isIterable(logPathList)):
        raise Exception("logPathList should be a list.")

    if labelList is None:
        labelList = ["Step {0}".format(i)
                     for i in range(1, len(logPathList) + 1)]
    else:
        if not(_isIterable(labelList)):
            raise Exception("labelList should be a list.")
        if len(labelList) != len(logPathList):
            raise Exception(
                "Number of labels should equal number of log files.")

    initialPercent = 1.0
    initialX = 0
    levelXList = []
    levelPercentList = []
    for (i, logPath) in enumerate(logPathList):
        percentList = imgMetamorphosisLogParser(logPath)[:, 1] * initialPercent
        numIterations = len(percentList)
        if useTime:
            # Parse run time from log and convert to minutes
            time = float(txtRead(logPath).split(
                "Time = ")[1].split("s ")[0]) / 60.0
            xList = np.linspace(0, time, numIterations + 1)[1:] + initialX
        else:
            xList = np.arange(0, numIterations) + initialX

        if not useLog:
            if i == 0:
                xList = np.array([initialX] + list(xList))
                percentList = np.array([initialPercent] + list(percentList))

        levelXList += [xList]
        levelPercentList += [percentList]

        initialPercent = percentList[-1]
        initialX = xList[-1]

    for i in range(len(levelXList)):
        if i > 0:
            xList = np.concatenate((levelXList[i - 1][-1:], levelXList[i]))
            percentList = np.concatenate(
                (levelPercentList[i - 1][-1:], levelPercentList[i]))

        else:
            xList = levelXList[i]
            percentList = levelPercentList[i]

        plt.plot(xList, percentList, label=labelList[i], linewidth=1.5)

    # Add plot annotations
    if useTime:
        plt.xlabel("Time (Minutes)")
    else:
        plt.xlabel("Iteration")

    plt.ylabel("Normalized $M(I(1), I_1)$")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if useLog:
        plt.xscale("log")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.autoscale(enable=True, axis='x', tight=True)

    # Fix maximum y to 1.0
    ylim = list(ax.get_ylim())
    ylim[1] = 1.0
    ax.set_ylim(ylim)


def imgMetamorphosisLogParser(logPath):
    logText = txtRead(logPath)
    lineList = logText.split("\n")

    for (lineIndex, line) in enumerate(lineList):
        if "E, E_velocity, E_rate, E_image" in line:
            break

    dataArray = np.empty((0, 5), float)
    for line in lineList[lineIndex:]:
        if "E =" in line:
            break

        try:
            (iterationString, dataString) = line.split(".\t")
        except BaseException:
            continue

        (energyString, velocityEnergyString, rateEnergyString,
         imageEnergyString, learningRateString) = (dataString.split(","))
        (energy, velocityEnergy, rateEnergy, learningRate) = map(float, [
            energyString, velocityEnergyString, rateEnergyString, learningRateString])
        (imageEnergy, imageEnergyPercent) = map(
            float, imageEnergyString.replace("(", "").replace("%)", "").split())

        imageEnergy = float(imageEnergyString.split(" (")[0])
        dataRow = np.array(
            [[energy, imageEnergyPercent / 100, velocityEnergy, learningRate, rateEnergy]])
        dataArray = np.concatenate((dataArray, dataRow), axis=0)

    return dataArray
