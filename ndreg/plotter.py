# plotter.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import util
import preprocessor

def imgShow(img, vmin=None, vmax=None, cmap=None, alpha=None,
            newFig=True, flip=None, numSlices=3, useNearest=False):
    """
    Displays an image.  Only 2D images are supported for now
    """
    if flip == None: flip = [0, 0, 0]
    if newFig: plt.figure()

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

            for (j, slice_) in enumerate(sliceList):
                sliceIndex = [0] * img.GetDimension()
                sliceIndex[2 - i] = int(slice_)
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

def imgSlices(img, flip=None, numSlices=1):
    if flip == None: flip = [0, 0, 0]
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

# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).
def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1,2,figsize=(10,8))
    # Draw the fixed image in the first subplot.
    plt.subplot(1,2,1)
    plt.imshow(fixed_npa[fixed_image_z,:,:],cmap=plt.cm.gray)
    plt.title('fixed image')
    plt.axis('off')
    # Draw the moving image in the second subplot.
    plt.subplot(1,2,2)

    plt.imshow(moving_npa[moving_image_z,:,:],cmap=plt.cm.gray)

    plt.title('moving image')
    plt.axis('off')
    
    plt.show()

# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space. 
from IPython.display import clear_output

def display_slices_with_alpha(fixed, moving, alpha, vmax):
    img = (1.0 - alpha)*fixed + alpha*moving

    plt.imshow(sitk.GetArrayViewFromImage(img),cmap=plt.cm.gray, vmax=vmax)

    plt.axis('off')
    plt.show()

def display_images_with_alpha(slice_num, alpha, fixed, moving, axis=2, vmax=1000):
    if axis == 0: display_slices_with_alpha(fixed[slice_num,:,:], moving[slice_num,:,:], alpha, vmax)
    elif axis == 1: display_slices_with_alpha(fixed[:,slice_num,:], moving[:,slice_num,:], alpha, vmax)
    else: display_slices_with_alpha(fixed[:,:,slice_num], moving[:,:,slice_num], alpha, vmax)
    
# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.    
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))  

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
