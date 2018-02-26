#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import util, registerer, preprocessor


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
    checker = imgChecker(defInImg, refImg)

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
    checker = imgChecker(defInImg, refImg)

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
