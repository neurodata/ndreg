import numpy as np
import tempfile
import SimpleITK as sitk
import shutil
import util
from itertools import product


def resample(image, transform, ref_img, default_value=0.0):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = ref_img
    interpolator = sitk.sitkBSpline
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)

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

def imgApplyAffine(inImg, affine, useNearest=False, size=[], spacing=[], origin=[0,0,0]):
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
        size = sizeOut(inImg, affine, spacing)
    else:
        if len(size) != inDimension:
            raise Exception(
                "size must have length {0}.".format(inDimension))

    # Apply affine transform
    outImg = sitk.Resample(inImg, size, affineTransform,
                           interpolator, origin, spacing)

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
        outTransform, sitk.sitkVectorFloat32, size, origin, spacing, direction)

def createTmpRegistration(inMask=None, refMask=None,
                          samplingFraction=1.0, dimension=3):
    identityTransform = sitk.Transform(3, sitk.sitkIdentity)
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
    inImg = util.imgCollapseDimension(inImg)
    refImg = util.imgCollapseDimension(refImg)
    if inMask:
        util.imgCollapseDimension(inMask)
    if refMask:
        util.imgCollapseDimension(refMask)

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
    inImg = util.imgCollapseDimension(inImg)
    refImg = util.imgCollapseDimension(refImg)
    if inMask:
        util.imgCollapseDimension(inMask)
    if refMask:
        util.imgCollapseDimension(refMask)
    tmpRegistration = createTmpRegistration(
        inMask, refMask, dimension=refImg.GetDimension(), samplingFraction=1.0)
    tmpRegistration.SetMetricAsMeanSquares()
    tmpRegistration.Execute(
        sitk.Cast(refImg, sitk.sitkFloat32), sitk.Cast(inImg, sitk.sitkFloat32))

    return tmpRegistration.GetMetricValue()