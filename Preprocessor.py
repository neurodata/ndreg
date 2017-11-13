import ndreg
import SimpleITK as sitk
import matplotlib
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

class Preprocessor:

    def __init__(self, img):
        if type(img) == sitk.SimpleITK.Image:
            self.img_sitk = img
            self.img_np = sitk.GetArrayFromImage(img)
            self.img_no_circle = None
            self.mask = None
        else: 
            raise Exception("Please convert your image into a SimpleITK image")

    def remove_streaks(self, radius=40):
        fft_img = np.fft.fftshift(np.fft.fft2(self.img_np, axes=(-2, -1)), axes=(-2, -1))
        center = [math.ceil((fft_img.shape[2]+1)/2), math.ceil((fft_img.shape[1]+1)/2)]
        fx, fy = np.meshgrid(np.arange(fft_img.shape[2]), np.arange(fft_img.shape[1]))
        fx = fx - center[0]
        fy = fy - center[1]
        r = np.sqrt(fx**2 + fy**2)
        theta = np.arctan2(fy, fx)
        
        streak1 = -90.0 * (math.pi/180.0) # radians
        streak2 = 100.0 * (math.pi/180.0) # radians
        streak3 = 80.0 * (math.pi/180.0) # radians
        streak4 = 90.0 * (math.pi/180.0) # radians
        streak5 = -80.0 * (math.pi/180.0) # radians
        streak6 = -100.0 * (math.pi/180.0) # radians
        delta = 2 * (math.pi/180.0) # radians
        small_num = 0.0

        streak_filter = np.ones(fft_img.shape[1:])
        
        # set streak regoins to small number
        streak_filter[(theta < streak1 + delta) & (theta > streak1 - delta)] = small_num
        streak_filter[(theta < streak2 + delta) & (theta > streak2 - delta)] = small_num
        streak_filter[(theta < streak3 + delta) & (theta > streak3 - delta)] = small_num
        streak_filter[(theta < streak4 + delta) & (theta > streak4 - delta)] = small_num
        streak_filter[(theta < streak5 + delta) & (theta > streak5 - delta)] = small_num
        streak_filter[(theta < streak6 + delta) & (theta > streak6 - delta)] = small_num
        streak_filter[ np.where(r < radius) ] = 1.0
        streak_filter_blurred = skimage.filters.gaussian(streak_filter, sigma=3)

        fft_filtered = np.ones(fft_img.shape, dtype=fft_img.dtype)
        for i in range(fft_img.shape[0]):
            fft_filtered[i,:,:] = np.multiply(fft_img[i,:,:], streak_filter_blurred)

        filtered_fft = np.fft.ifftshift(fft_filtered, axes=(-2,-1))
        img_streak_free = np.fft.ifft2(filtered_fft, axes=(-2,-1))
#         img_streak_free_sitk = sitk.GetImageFromArray(img_streak_free)
#         img_streak_free_sitk.SetSpacing(self.img_sitk.GetSpacing())
#         img_streak_free_sitk.SetDirection(self.img_sitk.GetDirection())
        return img_streak_free

    def create_mask(self, background_probability=0.75):
        if self.img_no_circle is None: self.remove_circle()
        gmix = GaussianMixture(n_components=3, covariance_type='full', init_params='kmeans', verbose=0)
        gmix.fit(self.img_no_circle.ravel().reshape(-1, 1))
        covariances = gmix.covariances_
        mean_background = gmix.means_.min()
        covariance_background = covariances[np.where( gmix.means_ == mean_background ) ][0][0]
        z_score = st.norm.ppf(background_probability)
        threshold = z_score * np.sqrt(covariance_background) + mean_background
        test_mask = (self.img_no_circle > threshold)
        eroded_im = morphology.opening(test_mask, selem=morphology.ball(2))
        connected_comp = skimage.measure.label(eroded_im)
        out = skimage.measure.regionprops(connected_comp)
        area_max = 0.0
        idx_max = 0
        for i in range(len(out)):
            if out[i].area > area_max:
                area_max = out[i].area
                idx_max = i+ 1
        connected_comp[ connected_comp != idx_max ] = 0
        self.mask = connected_comp
        return connected_comp
        
    def remove_circle(self, radius=170):
        origin_shift = [0, 0]
        center = [self.img_np.shape[-1]/2 + origin_shift[0], self.img_np.shape[-2]/2 + origin_shift[1]]
        x, y, z = self.img_np.shape
        mask = np.zeros((y,z))
        for i in range(y):
            for j in range(z):
                mask[i,j] = math.sqrt((i-center[1])**2 + (j-center[0])**2) 
        mask = (mask < radius).astype('uint64')
        new_mask = np.repeat(mask[None,:,:], self.img_np.shape[0], axis=0)
        img_masked = np.multiply(self.img_np, new_mask)
        self.img_no_circle = img_masked
        return img_masked
    
    def auto_preprocess(self):
        if self.mask is None: self.create_mask()
        masked_image = self.img_np.copy()
        masked_image[np.where(self.mask == 0)] = 0
        out_img = sitk.GetImageFromArray(masked_image)
        out_img.CopyInformation(self.img_sitk)
        return self.correct_bias_field(img=out_img)
       
    
    def correct_bias_field(self, img=None, mask=None, scale=0.2, numBins=128):
        """
        Bias corrects an image using the N4 algorithm
        """
        if img is None and self.img_no_circle is not None: 
            img = sitk.GetImageFromArray(self.img_no_circle)
            img.CopyInformation(self.img_sitk)
        elif img is None: img = self.img_sitk
        spacing = np.array(self.img_sitk.GetSpacing())/scale
        img_ds = ndreg.imgResample(img, spacing=spacing)

        # Calculate bias
        if mask is None:
            mask = sitk.Image(img_ds.GetSize(), sitk.sitkUInt8)+1
            mask.CopyInformation(img_ds)
        else:
            if type(mask) is not sitk.SimpleITK.Image:
                mask_sitk = sitk.GetImageFromArray(mask)
                mask_sitk.SetSpacing(self.img_sitk.GetSpacing())    
                mask = mask_sitk
            mask = ndreg.imgResample(mask, spacing=spacing)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        img_ds_bc = corrector.Execute(sitk.Cast(img_ds, sitk.sitkFloat32), mask)
        bias_ds = img_ds_bc - sitk.Cast(img_ds, img_ds_bc.GetPixelID())

        # Upsample bias
        bias = ndreg.imgResample(bias_ds, spacing=img.GetSpacing(), size=img.GetSize())

        # Apply bias to original image and threshold to eliminate negitive values
        upper = np.iinfo(sitkToNpDataTypes[self.img_sitk.GetPixelID()]).max
        img_bc = sitk.Threshold(img + sitk.Cast(bias, img.GetPixelID()),
                    lower=0, upper=upper)
        return img_bc 
