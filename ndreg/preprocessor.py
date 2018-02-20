import ndreg
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import filters, morphology
import sklearn
from sklearn.mixture.gaussian_mixture import GaussianMixture
import scipy.stats as st
import math

sitkToNpDataTypes = {sitk.sitkUInt8: np.uint8,
                     sitk.sitkUInt16: np.uint16,
                     sitk.sitkUInt32: np.uint32,
                     sitk.sitkInt8: np.int8,
                     sitk.sitkInt16: np.int16,
                     sitk.sitkInt32: np.int32,
                     sitk.sitkFloat32: np.float32,
                     sitk.sitkFloat64: np.float64,
                     }

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

def correct_bias_field(img, mask=None, scale=0.2, numBins=128, spline_order=4, niters=[50, 50, 50, 50],
                      num_control_pts=[5, 5, 5], fwhm=0.150, convergence_threshold=0.001):
    """
    Bias corrects an image using the N4 algorithm
    """
#     if type(img) is not sitk.SimpleITK.Image:
#         raise("Input image needs to be of type SimpleITK.SimpleITK.Image")

    spacing = np.array(img.GetSpacing())/scale
    img_ds = ndreg.imgResample(img, spacing=spacing)

    # Calculate bias
    if mask is None:
        mask = sitk.Image(img_ds.GetSize(), sitk.sitkUInt8)+1
        mask.CopyInformation(img_ds)
    else:
        if type(mask) is not sitk.SimpleITK.Image:
            mask_sitk = sitk.GetImageFromArray(mask)
            mask_sitk.CopyInformation(img)
            mask = mask_sitk
        mask = ndreg.imgResample(mask, spacing=spacing)

    img_ds_bc = sitk.N4BiasFieldCorrection(sitk.Cast(img_ds, sitk.sitkFloat32), mask, convergence_threshold,
                                           niters, splineOrder=spline_order,
                                           numberOfControlPoints=num_control_pts,
                                           biasFieldFullWidthAtHalfMaximum=fwhm)
    bias_ds = img_ds_bc / sitk.Cast(img_ds, img_ds_bc.GetPixelID())
    

    # Upsample bias
    bias = ndreg.imgResample(bias_ds, spacing=img.GetSpacing(), size=img.GetSize())

    img_bc = sitk.Cast(img, sitk.sitkFloat32) * sitk.Cast(bias, sitk.sitkFloat32)
    return img_bc

# TODO: finish this method
def create_iterative_mask(img):
    for i in range(7):
        out_mask = create_mask(img, use_triangle=True)
        ndreg.imgShow(sitk.GetImageFromArray(out_mask))
        
        img_bc_tmp = preprocessor.correct_bias_field(img, mask=out_mask, 
                                                     scale=0.25, spline_order=3, niters=[50, 50, 50, 50])
        img_bc_np = sitk.GetArrayFromImage(img_bc_tmp)
        plt.imshow(img_bc_np[80,:,:], vmax=10000)
        plt.colorbar()
        plt.show()
    
    mask_sitk = sitk.GetImageFromArray(out_mask)
    mask_sitk.CopyInformation(img)
    return 

# utility functions
def downsample(img, res=3):
    out_spacing = np.array(img.GetSpacing()) * (2.0**res)
    img_ds = skimage.measure.block_reduce(sitk.GetArrayViewFromImage(img),
                                          block_size=(2,2,2), func=np.mean)
    for i in range(res - 1):
        img_ds = skimage.measure.block_reduce(sitk.GetArrayViewFromImage(img_ds),
                                                     block_size=(2,2,2), func=np.mean)
    img_ds_sitk = sitk.GetImageFromArray(img_ds)
    img_ds_sitk.setSpacing(out_spacing)

def downsample_and_reorient(atlas, target, atlas_orient, target_orient, spacing, size=[], set_origin=True, dv_atlas=0.0, dv_target=0.0):
    """
    make sure img1 is the source and img2 is the target.
    iamges will be resampled to match the coordinate system of img2.
    """
    target_r = ndreg.imgReorient(target, target_orient, atlas_orient)
    size_atlas = atlas.GetSize()
    size_target = target_r.GetSize()
    dims_atlas = np.array(size_atlas)*np.array(atlas.GetSpacing())
    dims_target = np.array(size_target)*np.array(target_r.GetSpacing())
    max_size_per_dim = [max(dims_atlas[i], dims_target[i]) for i in range(len(dims_atlas))]
    print(max_size_per_dim)
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
   