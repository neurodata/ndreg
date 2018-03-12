#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import util, registerer, preprocessor

def register_brain(atlas, img, modality, outdir=None):
    """Register 3D mouse brain to the Allen Reference atlas using affine and deformable registration.
    
    Parameters:
    ----------
    atlas : {SimpleITK.SimpleITK.Image}
        Allen reference atlas or other atlas to register data to.
    img : {SimpleITK.SimpleITK.Image}
        Input observed 3D mouse brain volume
    modality : {str}
        Can be 'lavision' or 'colm' for either modality.
    outdir : {str}, optional
        Path to output directory to store intermediates. (the default is None, which will store all outputs in './')
    
    Returns
    -------
    SimpleITK.SimpleITK.Image
        The atlas deformed to fit the input image.
    """

    if outdir is None: outdir = './'
    final_transform = register_affine(sitk.Normalize(atlas),
                                                img,
                                                learning_rate=1e-1,
                                                grad_tol=4e-6,
                                                use_mi=False,
                                                iters=50,
                                                shrink_factors=[4,2,1],
                                                sigmas=[0.4, 0.2, 0.1],
                                                verbose=False)
    # save the affine transformation to outdir
    # make the dir if it doesn't exist
    util.dir_make(outdir)
    sitk.WriteTransform(final_transform, outdir + 'atlas_to_observed_affine.txt')
    atlas_affine = registerer.resample(atlas, final_transform, img, default_value=util.img_percentile(atlas,0.01))
    img_affine = registerer.resample(img, final_transform.GetInverse(), atlas, default_value=util.img_percentile(img,0.01))

    # whiten both images only before lddmm
    atlas_affine_w = sitk.AdaptiveHistogramEqualization(atlas_affine, [10,10,10], alpha=0.25, beta=0.25)
    img_w = sitk.AdaptiveHistogramEqualization(img, [10,10,10], alpha=0.25, beta=0.25)

    # then run lddmm
    e = 5e-3
    s = 0.1
    atlas_lddmm, field, inv_field = register_lddmm(sitk.Normalize(atlas_affine_w),
                                                                                                    sitk.Normalize(img_w),
                                                                                                    alpha_list=[0.05],
                                                                                                    scale_list = [0.0625, 0.125, 0.25, 0.5, 1.0],
                                                                                                    epsilon_list=e, sigma=s,
                                                                                                    min_epsilon_list=e*1e-6,
                                                                                                    use_mi=False, iterations=50, verbose=True,
                                                                                                    out_dir=outdir + 'lddmm')
    return atlas_lddmm

def register_affine(atlas, img, learning_rate=1e-2, iters=200, min_step=1e-10, shrink_factors=[1], sigmas=[.150], use_mi=False, grad_tol=1e-6, verbose=False):
    """
    Performs affine registration between an atlas an an image given that they have the same spacing.
    """
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
#     registration_method.SetMetricAsMeanSquares()
    if use_mi: registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=128)
    else: registration_method.SetMetricAsMeanSquares()
    # registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    # registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkBSpline)

    # Optimizer settings.
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=learning_rate,
                                                                 minStep=min_step,
    #                                                              estimateLearningRate=registration_method.EachIteration,
                                                                 gradientMagnitudeTolerance=grad_tol,
                                                                 numberOfIterations=iters)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=sigmas)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # initial transform
    initial_transform = sitk.AffineTransform(atlas.GetDimension())
    length = np.array(atlas.GetSize())*np.array(atlas.GetSpacing())
    initial_transform.SetCenter(length/2.0)

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform)

    # Connect all of the observers so that we can perform plotting during registration.
    if verbose:
        registration_method.AddCommand(sitk.sitkStartEvent, util.start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, util.end_plot)
        registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, util.update_multires_iterations)
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: util.plot_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(img, sitk.sitkFloat32),
                                                  sitk.Cast(atlas, sitk.sitkFloat32))
    return final_transform

def register_lddmm(affine_img, target_img, alpha_list=0.05, scale_list=[0.0625, 0.125, 0.25, 0.5, 1.0],
                   epsilon_list=1e-4, min_epsilon_list=1e-10, sigma=0.1, use_mi=False, iterations=200, inMask=None,
                   refMask=None, verbose=True, out_dir=''):
    if sigma == None:
        sigma = (0.1/target_img.GetNumberOfPixels())

    (field, invField) = registerer.imgMetamorphosisComposite(affine_img, target_img,
                                                                                                alphaList=alpha_list,
                                                                                                scaleList=scale_list,
                                                                                                epsilonList=epsilon_list,
                                                                                                minEpsilonList=min_epsilon_list,
                                                                                                sigma=sigma,
                                                                                                useMI=use_mi,
                                                                                                inMask=inMask,
                                                                                                refMask=refMask,
                                                                                                iterations=iterations,
                                                                                                verbose=verbose,
                                                                                                outDirPath=out_dir)

    source_lddmm = registerer.imgApplyField(affine_img, field,
                                            size=target_img.GetSize(),
                                            spacing=target_img.GetSpacing())
    return source_lddmm, field, invField

def register_rigid(atlas, img, learning_rate=1e-2, iters=200, min_step=1e-10, shrink_factors=[1], sigmas=[.150], use_mi=False, grad_tol=1e-6, verbose=False):
    """
    Performs affine registration between an atlas an an image given that they have the same spacing.
    """
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
#     registration_method.SetMetricAsMeanSquares()
    if use_mi: registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=128)
    else: registration_method.SetMetricAsMeanSquares()
    # registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    # registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkBSpline)

    # Optimizer settings.
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=learning_rate,
                                                                 minStep=min_step,
    #                                                              estimateLearningRate=registration_method.EachIteration,
                                                                 gradientMagnitudeTolerance=grad_tol,
                                                                 numberOfIterations=iters)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=sigmas)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # initial transform
    initial_transform = sitk.VersorRigid3DTransform()
    length = np.array(atlas.GetSize())*np.array(atlas.GetSpacing())
    initial_transform.SetCenter(length/2.0)

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform)

    # Connect all of the observers so that we can perform plotting during registration.
    if verbose:
        registration_method.AddCommand(sitk.sitkStartEvent, util.start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, util.end_plot)
        registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, util.update_multires_iterations)
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: util.plot_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(img, sitk.sitkFloat32),
                                                  sitk.Cast(atlas, sitk.sitkFloat32))
    return final_transform


def register_rigid_n_way(image_list,
                             epsilon = 1e-2,
                             medians = 10,
                             learning_rate=1e-2,
                             iters=200,
                             min_step=1e-10,
                             shrink_factors=[1],
                             sigmas=[.150],
                             use_mi=False,
                             grad_tol=1e-6,
                             verbose=False):

    """
    Register N 3-D images with rigid transformation.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3389460/

    1. Compute median of input images
    2. Register and resample images with median as atlas
    3. Repeat until transforms converge


    Parameters:
        n-way:
            image_list,    - input images (array of SimpleITK.SimpleITK.Image s)
            epsilon = 1e-2 - convergence threshold: MSE between consecutive median atlases
            medians = 10,  - maximum number of medians to calculate (upper limit for running time)

        register_rigid:
            learning_rate=1e-2,
            iters=200,
            min_step=1e-10,
            shrink_factors=[1],
            sigmas=[.150],
            use_mi=False,
            grad_tol=1e-6,
            verbose=False

    """

    depth = min([img.GetDepth() for img in image_list])
    source_images = [img[:,:,:depth] for img in image_list]

    atlas = sitk.GetImageFromArray(np.median([sitk.GetArrayFromImage(img) for img in image_list], axis=0))

    errors = []
    #init tranforms to identity and compose new transforms on top of it
    final_transforms = [sitk.Transform(sitk.TranslationTransform(3, (0,0,0))) for i in range(len(images))]

    #repeat until convergence
    for _ in range(medians):
        new_images = []
        for i in range(len(source_images)):
            print("Image {} of {}".format(i+1, len(source_images)))
            img = source_images[i]
            transform = registerer.register_rigid(atlas, img,
                                                        learning_rate=learning_rate,
                                                        iters=iters,
                                                        min_step=min_step,
                                                        shrink_factors=shrink_factors,
                                                        sigmas=sigmas,
                                                        use_mi=use_mi,
                                                        grad_tol=grad_tol,
                                                        verbose=verbose)
            new_images.append(registerer.resample(atlas, transform, img))
            final_transforms[i].AddTransform(transform)

        #images for next pass or output
        source_images = new_images
        new_atlas = sitk.GetImageFromArray(np.median(np.array(
            [sitk.GetArrayFromImage(img) for img in source_images]), axis=0))

        total_error = np.sum(calculate_error(atlas, new_atlas))
        errors.append(total_error)

        #Convergence check (early stopping)
        if total_error < epsilon:
            break
    return final_transforms, errors



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
    defInImg = registerer.imgApplyField(inImg, field, size=refImg.GetSize())
    checker = registerer.imgChecker(defInImg, refImg)

    sliceList = []
    for i in range(inImg.GetDimension()):
        step = [5] * inImg.GetDimension()
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

    if not(util.is_iterable(size)):
        raise Exception("size must be a list.")
    if not(util.is_iterable(spacing)):
        raise Exception("spacing must be a list.")
    if not(util.is_iterable(step)):
        raise Exception("step must be a list.")
    if len(size) != len(spacing):
        raise Exception("len(size) != len(spacing)")
    if len(size) != len(step):
        raise Exception("len(size) != len(step)")

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
        grid = registerer.imgApplyField(grid, field, size=size)

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
    defInImg = registerer.imgApplyField(inImg, field, size=refImg.GetSize())
    inImg = registerer.imgApplyAffine(
        inImg, [
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], size=refImg.GetSize())
    checker = registerer.imgChecker(defInImg, refImg)

    sliceList = []
    for i in range(inImg.GetDimension()):
        step = [20] * inImg.GetDimension()
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
    if not(util.is_iterable(logPathList)):
        raise Exception("logPathList should be a list.")

    if labelList is None:
        labelList = ["Step {0}".format(i)
                     for i in range(1, len(logPathList) + 1)]
    else:
        if not(util.is_iterable(labelList)):
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
            time = float(util.txt_read(logPath).split(
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
    logText = util.txt_read(logPath)
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

def imgChecker(inImg, refImg, useHM=True, pattern=None):
    """
    Checkerboards input image with reference image
    """
    inImg = sitk.Cast(inImg, refImg.GetPixelID())
    inSize = list(inImg.GetSize())
    refSize = list(refImg.GetSize())
    if pattern is None: pattern = [4]*inImg.GetDimension()

    if(inSize != refSize):
        sourceSize = np.array([inSize, refSize]).min(0)
        # Empty image with same size as reference image
        tmpImg = sitk.Image(refSize, refImg.GetPixelID())
        tmpImg.CopyInformation(refImg)
        inImg = sitk.Paste(tmpImg, inImg, sourceSize)

    if useHM:
        inImg = preprocessor.imgHM(inImg, refImg)

    return sitk.CheckerBoardImageFilter().Execute(inImg, refImg, pattern)

