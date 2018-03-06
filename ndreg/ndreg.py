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
    atlas_affine = registerer.resample(atlas, final_transform, img, default_value=imgPercentile(atlas,0.01))
    img_affine = registerer.resample(img, final_transform.GetInverse(), atlas, default_value=imgPercentile(img,0.01))

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
