import ndreg
import csv
import numpy as np
import tempfile
import SimpleITK as sitk
import skimage
import matplotlib
from matplotlib import pyplot as plt

import shutil
import util
import preprocessor
from itertools import product

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
        outDirPath = util.dir_make(outDirPath)

    inPath = outDirPath + "in.img"
    util.imgWrite(inImg, inPath)
    refPath = outDirPath + "ref.img"
    util.imgWrite(refImg, refPath)
    outPath = outDirPath + "out.img"

    fieldPath = outDirPath + "field.vtk"
    invFieldPath = outDirPath + "invField.vtk"

    binPath = util.ndregDirPath + "metamorphosis "
    steps = 5
    command = binPath + " --in {0} --ref {1} --out {2} --alpha {3} --beta {4} --field {5} --invfield {6} --iterations {7} --scale {8} --steps {9} --verbose ".format(
        inPath, refPath, outPath, alpha, beta, fieldPath, invFieldPath, iterations, scale, steps)
    if(not useBias):
        command += " --mu 0"
    if(useMI):
        command += " --cost 1 --sigma {}".format(sigma)
        if not(epsilon is None):
            command += " --epsilon {0}".format(epsilon)
        else:
            command += " --epsilon 1e-3"
    else:
        command += " --sigma {}".format(sigma)
        if not(epsilon is None):
            command += " --epsilon {0}".format(epsilon)
        else:
            command += " --epsilon 1e-3"

    if not(minEpsilon is None):
        command += " --epsilonmin {0}".format(minEpsilon)

    if(inMask):
        inMaskPath = outDirPath + "inMask.img"
        util.imgWrite(inMask, inMaskPath)
        command += " --inmask " + inMaskPath

    if(refMask):
        refMaskPath = outDirPath + "refMask.img"
        util.imgWrite(refMask, refMaskPath)
        command += " --refmask " + refMaskPath

    if debug:
        command = "/usr/bin/time -v " + command
        print(command)

    # os.system(command)
    (_, logText) = util.run_shell_command(command, verbose=verbose)

    logPath = outDirPath + "log.txt"
    util.txt_write(logText, logPath)

    field = util.imgRead(fieldPath)
    invField = util.imgRead(invFieldPath)

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
        outDirPath = util.dir_make(outDirPath)

    if util.is_number(alphaList):
        alphaList = [float(alphaList)]
    if util.is_number(betaList):
        betaList = [float(betaList)]
    if util.is_number(scaleList):
        scaleList = [float(scaleList)]

    numSteps = max(len(alphaList), len(betaList), len(scaleList))

    if util.is_number(epsilonList):
        epsilonList = [float(epsilonList)] * numSteps
    elif epsilonList is None:
        epsilonList = [None] * numSteps

    if util.is_number(minEpsilonList):
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
                util.imgWrite(compositeInvField, invFieldPath)
                util.imgWrite(compositeField, fieldPath)

        inImg = imgApplyField(origInImg, compositeField, size=refImg.GetSize())
        if(inMask):
            inMask = imgApplyField(origInMask,
                                   compositeField, size=refImg.GetSize(), useNearest=True)
        # vikram added this
#        if verbose: imgShow(inImg, vmax=imgPercentile(inImg, 0.99))

    # Write final results
    if outDirPath != "":
        util.imgWrite(compositeField, outDirPath + "field.vtk")
        util.imgWrite(compositeInvField, outDirPath + "invField.vtk")
        util.imgWrite(inImg, outDirPath + "out.img")
        util.imgWrite(imgChecker(inImg, refImg), outDirPath + "checker.img")

    if useTempDir:
        shutil.rmtree(outDirPath)
    return (compositeField, compositeInvField)

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