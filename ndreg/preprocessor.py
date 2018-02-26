import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import filters, morphology
import sklearn
from sklearn.mixture.gaussian_mixture import GaussianMixture
import scipy.stats as st
import math
import SimpleITK as sitk
import util

dimension = 3

def create_mask(img, background_probability=0.75, use_triangle=False, use_otsu=False):
    test_mask = None
    if use_triangle:
        test_mask = sitk.GetArrayFromImage(sitk.TriangleThreshold(img, 0, 1))
    elif use_otsu:
        test_mask = sitk.GetArrayFromImage(sitk.OtsuThreshold(img, 0, 1))
    else:
        if type(img) is sitk.SimpleITK.Image:
            img = sitk.GetArrayFromImage(img)
        gmix = GaussianMixture(n_components=3, covariance_type='full', init_params='kmeans', verbose=0)
        gmix.fit(img.ravel().reshape(-1, 1))
        covariances = gmix.covariances_
        mean_background = gmix.means_.min()
        covariance_background = covariances[np.where( gmix.means_ == mean_background ) ][0][0]
        z_score = st.norm.ppf(background_probability)
        threshold = z_score * np.sqrt(covariance_background) + mean_background
        test_mask = (img > threshold)
    eroded_im = morphology.opening(test_mask, selem=morphology.ball(2))
    connected_comp = skimage.measure.label(eroded_im)
    out = skimage.measure.regionprops(connected_comp)
    area_max = 0.0
    idx_max = 0
    for i in range(len(out)):
        if out[i].area > area_max:
            area_max = out[i].area
            idx_max = i+1
    connected_comp[ connected_comp != idx_max ] = 0
    mask = connected_comp
    mask_sitk = sitk.GetImageFromArray(mask)
    mask_sitk.CopyInformation(img)
    return mask_sitk

def imgHM(inImg, refImg, numMatchPoints=64, numBins=256):
    """
    Histogram matches input image to reference image and writes result to output image
    """
    inImg = sitk.Cast(inImg, refImg.GetPixelID())
    return sitk.HistogramMatchingImageFilter().Execute(
        inImg, refImg, numBins, numMatchPoints, False)

def correct_bias_field(img, mask=None, scale=0.2, numBins=128, spline_order=4, niters=[50, 50, 50, 50],
                      num_control_pts=[5, 5, 5], fwhm=0.150, convergence_threshold=0.001):
    """
    Bias corrects an image using the N4 algorithm
    """
     # do in case image has 0 intensities
    # add a small constant that depends on
    # distribution of intensities in the image
    stats = sitk.StatisticsImageFilter()
    stats.Execute(img)
    std = math.sqrt(stats.GetVariance())
    img_rescaled = sitk.Cast(img, sitk.sitkFloat32) + 0.1*std
    
    spacing = np.array(img_rescaled.GetSpacing())/scale
    img_ds = imgResample(img_rescaled, spacing=spacing)
    img_ds = sitk.Cast(img_ds, sitk.sitkFloat32)
    

    # Calculate bias
    if mask is None:
        mask = sitk.Image(img_ds.GetSize(), sitk.sitkUInt8)+1
        mask.CopyInformation(img_ds)
    else:
        if type(mask) is not sitk.SimpleITK.Image:
            mask_sitk = sitk.GetImageFromArray(mask)
            mask_sitk.CopyInformation(img)
            mask = mask_sitk
        mask = imgResample(mask, spacing=spacing)
    
    img_ds_bc = sitk.N4BiasFieldCorrection(img_ds, mask, convergence_threshold,
                                           niters, splineOrder=spline_order,
                                           numberOfControlPoints=num_control_pts,
                                           biasFieldFullWidthAtHalfMaximum=fwhm)
    bias_ds = img_ds_bc / sitk.Cast(img_ds, img_ds_bc.GetPixelID())
    

    # Upsample bias
    bias = imgResample(bias_ds, spacing=img.GetSpacing(), size=img.GetSize())

    img_bc = sitk.Cast(img, sitk.sitkFloat32) * sitk.Cast(bias, sitk.sitkFloat32)
    return img_bc, bias

# TODO: finish this method
#def create_iterative_mask(img):
#    for i in range(7):
#        out_mask = create_mask(img, use_triangle=True)
#        imgShow(sitk.GetImageFromArray(out_mask))
#        
#        img_bc_tmp = preprocessor.correct_bias_field(img, mask=out_mask, 
#                                                     scale=0.25, spline_order=3, niters=[50, 50, 50, 50])
#        img_bc_np = sitk.GetArrayFromImage(img_bc_tmp)
#        plt.imshow(img_bc_np[80,:,:], vmax=10000)
#        plt.colorbar()
#        plt.show()
#    
#    mask_sitk = sitk.GetImageFromArray(out_mask)
#    mask_sitk.CopyInformation(img)
#    return 
#
# utility functions
def downsample(img, res=3):
    out_spacing = np.array(img.GetSpacing()) * (2.0**res)
    img_ds = skimage.measure.block_reduce(sitk.GetArrayViewFromImage(img),
                                          block_size=(2,2,2), func=np.mean)
    for _ in range(res - 1):
        img_ds = skimage.measure.block_reduce(sitk.GetArrayViewFromImage(img_ds),
                                                     block_size=(2,2,2), func=np.mean)
    img_ds_sitk = sitk.GetImageFromArray(img_ds)
    img_ds_sitk.setSpacing(out_spacing)

def downsample_and_reorient(atlas, target, atlas_orient, target_orient, spacing, size=[], set_origin=True, dv_atlas=0.0, dv_target=0.0):
    """
    make sure img1 is the source and img2 is the target.
    iamges will be resampled to match the coordinate system of img2.
    """
    target_r = imgReorient(target, target_orient, atlas_orient)
    size_atlas = atlas.GetSize()
    size_target = target_r.GetSize()
    dims_atlas = np.array(size_atlas)*np.array(atlas.GetSpacing())
    dims_target = np.array(size_target)*np.array(target_r.GetSpacing())
    max_size_per_dim = [max(dims_atlas[i], dims_target[i]) for i in range(len(dims_atlas))]
    vox_sizes = [int(i) for i in (np.array(max_size_per_dim) / spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetSize(vox_sizes)
    resampler.SetOutputSpacing([spacing]*3)
    resampler.SetDefaultPixelValue(dv_target)

    out_target = resampler.Execute(target_r)
    resampler.SetDefaultPixelValue(dv_atlas)
    out_atlas = resampler.Execute(atlas)
    
    assert(out_target.GetOrigin() == out_atlas.GetOrigin())
    assert(out_target.GetSize() == out_atlas.GetSize())
    assert(out_target.GetSpacing() == out_atlas.GetSpacing())
    return out_atlas, out_target
   
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
    return outImg

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
#    if util.is_number(padding):
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

def preprocess_brain(img, spacing, modality, image_orientation, atlas_orientation='pir'):
    img = imgResample(img, spacing)
    mask_dilation_radius = 10 # voxels
    mask = sitk.BinaryDilate(create_mask(img, use_triangle=True), mask_dilation_radius)
    if modality.lower() == 'colm': mask = None
    img_bc = correct_bias_field(img, scale=0.25, spline_order=4, mask=mask,
                                                num_control_pts=[5,5,5],
                                                niters=[500, 500, 500, 500])
    img_bc = imgReorient(img_bc, image_orientation, atlas_orientation)
    img_bc_n = sitk.Normalize(img_bc)
    return img_bc_n
