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
from itertools import product
from landmarks import *

from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *
import requests
from requests import HTTPError

requests.packages.urllib3.disable_warnings()  # Disable InsecureRequestWarning
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
ndregTranslation = 0
ndregScale = 1
ndregRigid = 2  # 1
ndregAffine = 3  # 2


def isIterable(variable):
    """
    Returns True if variable is a list, tuple or any other iterable object
    """
    return hasattr(variable, '__iter__')


def isNumber(variable):
    """
    Returns True if varible is is a number
    """
    try:
        float(variable)
    except TypeError:
        return False
    return True


def isInteger(n, epsilon=1e-6):
    """
    Returns True if n is integer within error epsilon
    """
    return (n - int(n)) < epsilon


def run(command, checkReturnValue=True, verbose=False):
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
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]
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


#def imgZoom(img, point, size, spacing=[], useNearest=False, outsideValue=0):
#    """
#    Returns image of given size centered at given point at resolution given by spacing
#    """
#    if len(point) != img.GetDimension():
#        raise Exception(
#            "len(point) != " + str(img.GetDimension()))
#    if len(size) != img.GetDimension():
#        raise Exception(
#            "len(size) != " + str(img.GetDimension()))
#    if spacing == []:
#        spacing = img.GetSpacing()
#    else:
#        if len(spacing) != img.GetDimension():
#            raise Exception(
#                "len(spacing) != " + str(img.GetDimension()))
#
#    origin = np.array(point) - np.array(size) * np.array(spacing) * 0.5
#    return imgResample(img, spacing, size, useNearest, origin, outsideValue)
#
#
#def imgPad(img, padding=0, useNearest=False):
#    """
#    Pads image by given ammount of padding in units spacing.
#    For example if the input image has a voxel spacing of 0.5 and the padding=2.0 then the image will be padded by 4 voxels.
#    If the padding < 0 then the filter crops the image
#    """
#    if isNumber(padding):
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
#
#def imgLargestMaskObject(maskImg):
#    ccFilter = sitk.ConnectedComponentImageFilter()
#    labelImg = ccFilter.Execute(maskImg)
#    numberOfLabels = ccFilter.GetObjectCount()
#    labelArray = sitk.GetArrayFromImage(labelImg)
#    labelSizes = np.bincount(labelArray.flatten())
#    largestLabel = np.argmax(labelSizes[1:]) + 1
#    outImg = sitk.GetImageFromArray(
#        (labelArray == largestLabel).astype(np.int16))
#    # output image should have same metadata as input mask image
#    outImg.CopyInformation(maskImg)
#    return outImg
#
#
#def createTmpRegistration(inMask=None, refMask=None,
#                          samplingFraction=1.0, dimension=dimension):
#    identityTransform = sitk.Transform(dimension, sitk.sitkIdentity)
#    tmpRegistration = sitk.ImageRegistrationMethod()
#    tmpRegistration.SetInterpolator(sitk.sitkNearestNeighbor)
#    tmpRegistration.SetInitialTransform(identityTransform)
#    tmpRegistration.SetOptimizerAsGradientDescent(
#        learningRate=1e-14, numberOfIterations=1)
#    if samplingFraction != 1.0:
#        tmpRegistration.SetMetricSamplingPercentage(samplingFraction)
#        tmpRegistration.SetMetricSamplingPercentage(tmpRegistration.RANDOM)
#
#    if(inMask):
#        tmpRegistration.SetMetricMovingMask(inMask)
#    if(refMask):
#        tmpRregistration.SetMetricFixedMask(refMask)
#
#    return tmpRegistration
#

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


def imgMakeRGBA(imgList, dtype=sitk.sitkUInt8):
    if len(imgList) < 3 or len(imgList) > 4:
        raise Exception(
            "imgList must contain 3 ([r,g,b]) or 4 ([r,g,b,a]) channels.")

    inDatatype = sitkToNpDataTypes[imgList[0].GetPixelID()]
    outDatatype = sitkToNpDataTypes[dtype]
    inMin = np.iinfo(inDatatype).min
    inMax = np.iinfo(inDatatype).max
    outMin = np.iinfo(outDatatype).min
    outMax = np.iinfo(outDatatype).max

    castImgList = []
    for img in imgList:
        castImg = sitk.Cast(sitk.IntensityWindowingImageFilter().Execute(
            img, inMin, inMax, outMin, outMax), dtype)
        castImgList.append(castImg)

    if len(imgList) == 3:
        channelSize = list(imgList[0].GetSize())
        alphaArray = outMax * np.ones(channelSize[::-1], dtype=outDatatype)
        alphaChannel = sitk.GetImageFromArray(alphaArray)
        alphaChannel.CopyInformation(imgList[0])
        castImgList.append(alphaChannel)

    return sitk.ComposeImageFilter().Execute(castImgList)


def imgThreshold(img, threshold=0):
    """
    Thresholds image at inPath at given threshold and writes result to outPath.
    """
    return sitk.BinaryThreshold(img, 0, threshold, 0, 1)


#def imgMakeMask(inImg, threshold=None, forgroundValue=1):
#    """
#    Generates morphologically smooth mask with given forground value from input image.
#    If a threshold is given, the binary mask is initialzed using the given threshold...
#    ...Otherwise it is initialized using Otsu's Method.
#    """
#
#    if threshold is None:
#        # Initialize binary mask using otsu threshold
#        inMask = sitk.BinaryThreshold(
#            inImg, 0, 0, 0, forgroundValue)  # Mask of non-zero voxels
#        otsuThresholder = sitk.OtsuThresholdImageFilter()
#        otsuThresholder.SetInsideValue(0)
#        otsuThresholder.SetOutsideValue(forgroundValue)
#        otsuThresholder.SetMaskValue(forgroundValue)
#        tmpMask = otsuThresholder.Execute(inImg, inMask)
#    else:
#        # initialzie binary mask using given threshold
#        tmpMask = sitk.BinaryThreshold(inImg, 0, threshold, 0, forgroundValue)
#
#    # Assuming input image is has isotropic resolution...
#    # ... compute size of morphological kernels in voxels.
#    spacing = min(list(inImg.GetSpacing()))
#    openingRadiusMM = 0.05  # In mm
#    closingRadiusMM = 0.2   # In mm
#    openingRadius = max(1, int(round(openingRadiusMM / spacing)))  # In voxels
#    closingRadius = max(1, int(round(closingRadiusMM / spacing)))  # In voxels
#
#    # Morphological open mask remove small background objects
#    opener = sitk.GrayscaleMorphologicalOpeningImageFilter()
#    opener.SetKernelType(sitk.sitkBall)
#    opener.SetKernelRadius(openingRadius)
#    tmpMask = opener.Execute(tmpMask)
#
#    # Morphologically close mask to fill in any holes
#    closer = sitk.GrayscaleMorphologicalClosingImageFilter()
#    closer.SetKernelType(sitk.sitkBall)
#    closer.SetKernelRadius(closingRadius)
#    outMask = closer.Execute(tmpMask)
#
#    return imgLargestMaskObject(outMask)
#

def imgMask(img, mask):
    """
    Convenience function to apply mask to image
    """
    mask = imgResample(mask, img.GetSpacing(), img.GetSize(), useNearest=True)
    mask = sitk.Cast(mask, img.GetPixelID())
    return sitk.MaskImageFilter().Execute(img, mask)


def sizeOut(inImg, transform, outSpacing):
    """
    Calculates size of bounding box which encloses transformed image
    """
    outCornerPointList = []
    inSize = inImg.GetSize()
    for corner in product((0, 1), repeat=inImg.GetDimension()):
        inCornerIndex = np.array(corner) * np.array(inSize)
        inCornerPoint = inImg.TransformIndexToPhysicalPoint(inCornerIndex)
        outCornerPoint = transform.GetInverse().TransformPoint(inCornerPoint)
        outCornerPointList += [list(outCornerPoint)]

    size = np.ceil(np.array(outCornerPointList).max(
        0) / outSpacing).astype(int)
    return size


def affineToField(affine, size, spacing):
    """
    Generates displacement field with given size and spacing based on affine parameters.
    """
    if len(size) != dimension:
        raise Exception(
            "size must have length {0}.".format(dimension))
    if len(spacing) != dimension:
        raise Exception(
            "spacing must have length {0}.".format(dimension))

    # Set affine parameters
    affineTransform = sitk.AffineTransform(dimension)
    numParameters = len(affineTransform.GetParameters())
    if len(affine) != numParameters:
        raise Exception(
            "affine must have length {0}.".format(numParameters))
    affineTransform = sitk.AffineTransform(dimension)
    affineTransform.SetParameters(affine)

    # Convert affine transform to field
    return sitk.TransformToDisplacementFieldFilter().Execute(
        affineTransform, vectorType, size, zeroOrigin, spacing, identityDirection)


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


def affineInverse(affine):
    # x0 = A0*x1 + b0
    # x1 = (A0.I)*x0 + (-A0.I*b0) = A1*x0 + b1

    if not isIterable(affine):
        raise Exception("affine must be a list.")
    numParameters = len(affine)
    n = 0.5 * (-1 + math.sqrt(1 + 4 * numParameters))  # dimension
    if not isInteger(n):
        raise Exception(
            "Must have len(affine) = n*n + n where n is some interger.")
    n = int(n)

    A0 = np.mat(affine[:n * n]).reshape(n, n)
    b0 = np.mat(affine[n * n:]).reshape(n, 1)

    A1 = A0.I
    b1 = -A1 * b0
    return A1.flatten().tolist()[0] + b1.flatten().tolist()[0]


def affineRotate(theta, center=[], useDegrees=True):
    cos = np.cos
    sin = np.sin

    theta = np.array(theta)
    if useDegrees:
        theta = theta * (np.pi / 180)

    if len(theta) == 1:
        n = 2  # dimension
    elif len(theta) == 3:
        n = len(theta)
    else:
        raise Exception("theta has invalid dimension")

    if center == []:
        center = [0] * n
    else:
        if not isIterable(center):
            raise Exception("center must be a list.")
        if len(center) != n:
            raise Exception("center has invalid dimension.")

    center = np.mat(center).transpose()

    if n == 2:
        theta = theta[0]
        A = np.mat([[cos(theta), -sin(theta)],
                    [sin(theta), cos(theta)]])
    elif n == 3:
        A_x = np.mat([[1, 0, 0],
                      [0, cos(theta[0]), -sin(theta[0])],
                      [0, sin(theta[0]), cos(theta[0])]])
        A_y = np.mat([[cos(theta[1]), 0, sin(theta[1])],
                      [0, 1, 0],
                      [-sin(theta[1]), 0, cos(theta[1])]])
        A_z = np.mat([[cos(theta[2]), -sin(theta[2]), 0],
                      [sin(theta[2]), cos(theta[2]), 0],
                      [0, 0, 1]])
        A = A_z * A_y * A_x

    b = -A * center + center
    return A.flatten().tolist()[0] + b.flatten().tolist()[0]


def affineApplyAffine(inAffine, affine):
    """ A_{outAffine} = A_{inAffine} \circ A_{affine} """
    if (not(isIterable(inAffine)) or not(isIterable(affine))
            ):
        raise Exception("inAffine and affine must be lists.")
    numParameters = len(inAffine)
    n = 0.5 * (-1 + math.sqrt(1 + 4 * numParameters))  # dimension
    if not isInteger(n):
        raise Exception(
            "Must have len(inAffine) = n*n + n where n is some interger.")
    if numParameters != len(affine):
        raise Exception(
            "inAffine and affine should be lists of same length.")

    n = int(n)
    A0 = np.array(affine[:n * n]).reshape(n, n)
    b0 = np.array(affine[n * n:]).reshape(n, 1)
    A1 = np.array(inAffine[:n * n]).reshape(n, n)
    b1 = np.array(inAffine[n * n:]).reshape(n, 1)

    # x0 = A0*x1 + b0
    # x1 = A1*x2 + b1
    # x0 = A0*(A1*x2 + b1) + b0 = (A0*A1)*x2 + (A0*b1 + b0)
    A = np.dot(A0, A1)
    b = np.dot(A0, b1) + b0

    outAffine = A.flatten().tolist() + b.flatten().tolist()
    return outAffine


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

def imgAffine(inImg, refImg, method=ndregAffine, scale=1.0, useNearest=False, useMI=False, numBins=32, iterations=1000, inMask=None, refMask=None, verbose=False, epsilon=0.1):
    """
    Perform Affine Registration between input image and reference image
    """
    inDimension = inImg.GetDimension()

    # Rescale images
    refSpacing = refImg.GetSpacing()
    spacing = [x / scale for x in refSpacing]
    inImg = imgResample(inImg, spacing, useNearest=useNearest)
    refImg = imgResample(refImg, spacing, useNearest=useNearest)
    if(inMask): imgResample(inMask, spacing, useNearest=False)
    if(refMask): imgResample(refMask, spacing, useNearest=False)
    
    # Set interpolator
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]
    
    # Set transform
    refCenter = refImg.TransformIndexToPhysicalPoint(np.array(refImg.GetSize())/2)
        
    try:
        rigidTransformList = [sitk.Similarity2DTransform(), sitk.Similarity3DTransform()]
        transform = [sitk.TranslationTransform(inDimension), sitk.ScaleTransform(inDimension), rigidTransformList[inDimension-2], sitk.AffineTransform(inDimension)][method]
        if method > ndregTranslation: transform.SetCenter(refCenter)
    except:
        raise Exception("method is invalid")
    
    # Do registration
    registration = sitk.ImageRegistrationMethod()
    registration.SetInterpolator(interpolator)
    registration.SetInitialTransform(transform)

    if(inMask): registration.SetMetricMovingMask(inMask)
    if(refMask): registration.SetMetricFixedMask(refMask)
    
    if useMI:
        registration.SetMetricAsMattesMutualInformation(numBins)
 
    else:
        registration.SetMetricAsMeanSquares()

    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=epsilon, numberOfIterations=iterations, estimateLearningRate=registration.EachIteration,minStep=0.001)
    if(verbose): registration.AddCommand(sitk.sitkIterationEvent, lambda : print("{0}.\t {1}".format(registration.GetOptimizerIteration(),registration.GetMetricValue())))

    numParameters = len(transform.GetParameters())
    
    optimizerScales = [1.0]*numParameters
    translationScale = 3e-3
    if method == ndregRigid:
        optimizerScales[inDimension:2*inDimension] = [translationScale]*inDimension
    elif method == ndregAffine: 
        optimizerScales[inDimension**2:] = [translationScale]*inDimension
    
    registration.SetOptimizerScales(optimizerScales)

    sigma = np.mean(np.array(refImg.GetSize())*np.array(refImg.GetSpacing())*0.02)
    registration.Execute(sitk.SmoothingRecursiveGaussian(refImg,sigma),
                         sitk.SmoothingRecursiveGaussian(inImg,sigma) )


    idAffine = list(sitk.AffineTransform(inDimension).GetParameters())    

    if method > ndregTranslation:
        try:
            t = transform.GetTranslation()
        except:
            t = [0]*inDimension
        #affine = list(transform.GetMatrix()) + list(transform.GetTranslation())
        A = np.mat(transform.GetMatrix()).reshape(inDimension, inDimension)
        c = np.mat(refCenter).transpose()
        b = np.mat(t).transpose()
        offset = b+c - A*c
        affine = list(transform.GetMatrix()) + offset.flatten().tolist()[0]
    else:
        affine = idAffine[0:inDimension**2] + list(transform.GetOffset())
        
    return affine

def imgBC(img, mask=None, scale=1.0, numBins=64):
    """
    Bias corrects an image using the N4 algorithm
    """
    spacing = np.array(img.GetSpacing())/0.2
    img_ds = imgResample(img, spacing=spacing)

    # Calculate bias
    if mask is None:
        mask = sitk.Image(img_ds.GetSize(), sitk.sitkUInt8)+1
        mask.CopyInformation(img_ds)

    img_ds_bc = sitk.N4BiasFieldCorrection(sitk.Cast(img_ds, sitk.sitkFloat32), mask, numberOfHistogramBins=numBins)
    bias_ds = img_ds_bc - sitk.Cast(img_ds,img_ds_bc.GetPixelID())
    # Upsample bias    
    bias = imgResample(bias_ds, spacing=img.GetSpacing(), size=img.GetSize())

    # Apply bias to original image and threshold to eliminate negitive values
    upper = np.iinfo(sitkToNpDataTypes[img.GetPixelID()]).max
    img_bc = sitk.Threshold(img + sitk.Cast(bias, img.GetPixelID()),
                            lower=0,
                            upper=upper)
    return img_bc

def imgAffineComposite(inImg, refImg, scale=1.0, useNearest=False, useMI=False, iterations=1000, epsilon=0.1, inAffine=None, methodList=[
                       ndregTranslation, ndregScale, ndregRigid, ndregAffine], verbose=False, inMask=None, refMask=None, outDirPath=""):
    if outDirPath != "":
        outDirPath = dirMake(outDirPath)
    methodNameList = ["translation", "scale", "rigid", "affine"]
    origInImg = inImg
    origInMask = inMask
    origRefMask = refMask

    # initilize using input affine
    inDimension = inImg.GetDimension()
    if inAffine is None:
        inAffine = list(
            sitk.AffineTransform(inDimension).GetParameters())

    compositeAffine = inAffine
    inImg = imgApplyAffine(origInImg, compositeAffine)
    if(inMask):
        inMask = imgApplyAffine(
            origInMask, compositeAffine, useNearest=True)

    if outDirPath != "":
        imgWrite(inImg, outDirPath + "0_initial/in.img")
        if(inMask):
            imgWrite(inMask, outDirPath + "0_initial/inMask.img")
        txtWriteList(compositeAffine, outDirPath + "0_initial/affine.txt")

    # methodList = [ndregTranslation, ndregScale, ndregAffine]
    # methodNameList = ["translation", "scale", "affine"]

    for (step, method) in enumerate(methodList):
        methodName = methodNameList[method]
        stepDirPath = outDirPath + str(step + 1) + "_" + methodName + "/"
        if outDirPath != "":
            dirMake(stepDirPath)
        if(verbose):
            print("Step {0}:".format(methodName))

        affine = imgAffine(inImg, refImg, method=method, scale=scale, useNearest=useNearest, useMI=useMI,
                           iterations=iterations, epsilon=epsilon, inMask=inMask, refMask=refMask, verbose=verbose)
        compositeAffine = affineApplyAffine(affine, compositeAffine)

        inImg = imgApplyAffine(
            origInImg, compositeAffine, size=refImg.GetSize())
        if(inMask):
            inMask = imgApplyAffine(origInMask,
                                    compositeAffine, size=refImg.GetSize(), useNearest=False)

        if outDirPath != "":
            imgWrite(inImg, stepDirPath + "in.img")
            if(inMask):
                imgWrite(inMask, stepDirPath + "inMask.img")
            txtWriteList(compositeAffine, stepDirPath + "affine.txt")

    # Write final results
    if outDirPath != "":
        txtWrite(compositeAffine, outDirPath + "affine.txt")
        imgWrite(inImg, outDirPath + "out.img")
        imgWrite(imgChecker(inImg, refImg), outDirPath + "checker.img")

    return compositeAffine

def imgMetamorphosis(inImg, refImg, alpha=0.02, beta=0.05, scale=1.0, iterations=1000, epsilon=None, sigma=1e-4, useNearest=False,
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


def imgMetamorphosisComposite(inImg, refImg, alphaList=0.02, betaList=0.05, scaleList=1.0, iterations=1000, epsilonList=None, sigma=1e-4,
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

    if isNumber(alphaList):
        alphaList = [float(alphaList)]
    if isNumber(betaList):
        betaList = [float(betaList)]
    if isNumber(scaleList):
        scaleList = [float(scaleList)]

    numSteps = max(len(alphaList), len(betaList), len(scaleList))

    if isNumber(epsilonList):
        epsilonList = [float(epsilonList)] * numSteps
    elif epsilonList is None:
        epsilonList = [None] * numSteps

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
        imgShow(inImg, vmax=imgPercentile(inImg, 0.99))

    # Write final results
    if outDirPath != "":
        imgWrite(compositeField, outDirPath + "field.vtk")
        imgWrite(compositeInvField, outDirPath + "invField.vtk")
        imgWrite(inImg, outDirPath + "out.img")
        imgWrite(imgChecker(inImg, refImg), outDirPath + "checker.img")

    if useTempDir:
        shutil.rmtree(outDirPath)
    return (compositeField, compositeInvField)

def imgRegistration(inImg, refImg, scale=1.0, affineScale=1.0, lddmmScaleList=[1.0], lddmmAlphaList=[
                    0.02], iterations=1000, useMI=False, useNearest=True, inAffine=identityAffine, padding=0, inMask=None, refMask=None, verbose=False, outDirPath=""):
    if outDirPath != "":
        outDirPath = dirMake(outDirPath)

    initialDirPath = outDirPath + "0_initial/"
    affineDirPath = outDirPath + "1_affine/"
    lddmmDirPath = outDirPath + "2_lddmm/"
    origInImg = inImg
    origRefImg = refImg

    # Resample and histogram match in and ref images
    refSpacing = refImg.GetSpacing()
    spacing = [x / scale for x in refSpacing]
    inImg = imgPad(imgResample(
        inImg, spacing, useNearest=useNearest), padding, useNearest)
    refImg = imgPad(imgResample(refImg, spacing,
                                useNearest=useNearest), padding, useNearest)
    if(inMask):
        inMask = imgPad(imgResample(
            inMask, spacing, useNearest=True), padding, True)
    if(refMask):
        refMask = imgPad(imgResample(
            refMask, spacing, useNearest=True), padding, True)

    if not useMI:
        inImg = imgHM(inImg, refImg)
    initialInImg = inImg
    initialInMask = inMask
    initialRefMask = refMask
    if outDirPath != "":
        imgWrite(inImg, initialDirPath + "in.img")
        imgWrite(refImg, initialDirPath + "ref.img")
        if(inMask):
            imgWrite(inMask, initialDirPath + "inMask.img")
        if(refMask):
            imgWrite(refMask, initialDirPath + "refMask.img")

    if(verbose):
        print("Affine alignment")
    affine = imgAffineComposite(inImg, refImg, scale=affineScale, useMI=useMI, iterations=iterations,
                                inAffine=inAffine, verbose=verbose, inMask=inMask, refMask=refMask, outDirPath=affineDirPath)
    affineField = affineToField(affine, refImg.GetSize(), refImg.GetSpacing())
    invAffine = affineInverse(affine)
    invAffineField = affineToField(
        invAffine, inImg.GetSize(), inImg.GetSpacing())
    inImg = imgApplyField(initialInImg, affineField, size=refImg.GetSize())
    if(inMask):
        inMask = imgApplyField(initialInMask,
                               affineField, size=refImg.GetSize(), useNearest=True)
    if(refMask):
        refMask = imgApplyField(initialRefMask,
                                affineField, size=refImg.GetSize(), useNearest=True)

    if outDirPath != "":
        imgWrite(inImg, affineDirPath + "in.img")
        if(inMask):
            imgWrite(inMask, affineDirPath + "inMask.img")
        if(refMask):
            imgWrite(refMask, affineDirPath + "refMask.img")

    # Deformably align in and ref images
    if(verbose):
        print("Deformable alignment")
    (field, invField) = imgMetamorphosisComposite(inImg, refImg, alphaList=lddmmAlphaList, scaleList=lddmmScaleList,
                                                  useBias=False, useMI=useMI, verbose=verbose, iterations=iterations, inMask=inMask, refMask=refMask, outDirPath=lddmmDirPath)

    field = fieldApplyField(field, affineField)
    invField = fieldApplyField(invAffineField, invField)
    inImg = imgApplyField(initialInImg, field, size=refImg.GetSize())

    if outDirPath != "":
        imgWrite(field, lddmmDirPath + "field.vtk")
        imgWrite(invField, lddmmDirPath + "invField.vtk")
        imgWrite(inImg, lddmmDirPath + "in.img")
        imgWrite(imgChecker(inImg, refImg), lddmmDirPath + "checker.img")

    # Remove padding from fields
    field = imgPad(field, -padding)
    invField = imgPad(invField, -padding)

    if outDirPath != "":
        imgWrite(field, outDirPath + "field.vtk")
        imgWrite(invField, outDirPath + "invField.vtk")

    return (field, invField)


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

    if not(isIterable(size)):
        raise Exception("size must be a list.")
    if not(isIterable(spacing)):
        raise Exception("spacing must be a list.")
    if not(isIterable(step)):
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
    if not(isIterable(logPathList)):
        raise Exception("logPathList should be a list.")

    if labelList is None:
        labelList = ["Step {0}".format(i)
                     for i in range(1, len(logPathList) + 1)]
    else:
        if not(isIterable(labelList)):
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


def lmkApplyField(inLmk, field, spacing=[1, 1, 1]):
    # Create transform
    transform = sitk.DisplacementFieldTransform(dimension)
    transform.SetInterpolator(sitk.sitkLinear)
    transform.SetDisplacementField(sitk.Cast(field, sitk.sitkVectorFloat64))

    outLmkList = []
    for lmk in inLmk.GetLandmarks():
        name = lmk[0]
        inPoint = lmk[1:]
        outPoint = transform.TransformPoint(inPoint)
        outLmkList += [[name] + list(outPoint)]

    return landmarks(outLmkList, inLmk.spacing)
