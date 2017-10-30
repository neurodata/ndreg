import ndreg
import SimpleITK as sitk


class Registerer:

    def __init__(self, img, atlas):
        if type(img) != sitk.SimpleITK.Image:
            raise Exception("Please convert your image into a SimpleITK Image")
        if type(atlas) != sitk.SimpleITK.Image:
            raise Exception("Please convert your image into a SimpleITK Image")
        else:
            self.img = img
            self.atlas = atlas
            self.affine = []
            self.atlas_affine = None
            self.fieldComposite = None
            self.invFieldComposite = None

    def register_affine(self, inOrient, refOrient, spacing, iterations=2000.0, resolutions=8.0):
        img_reoriented = ndreg.imgReorient(self.img, inOrient, refOrient)
        atlas_ds = sitk.Clamp(ndreg.imgResample(self.atlas, spacing), upperBound=ndreg.imgPercentile(self.atlas, 0.99))
        img_ds = sitk.Clamp(ndreg.imgResample(img_reoriented, spacing), upperBound=ndreg.imgPercentile(img_reoriented, 0.99))
        # normalize
        max_val = ndreg.imgPercentile(img_ds, 0.999)
        min_val = ndreg.imgPercentile(img_ds, 0.001)
        img_ds = (img_ds - min_val)/(max_val - min_val)
        
        max_val = ndreg.imgPercentile(atlas_ds, 0.999)
        min_val = ndreg.imgPercentile(atlas_ds, 0.001)
        atlas_ds = (atlas_ds - min_val)/(max_val - min_val)
        
        fixedImage = img_ds
        movingImage = atlas_ds
        
        # set parameters
        affineParameterMap = sitk.GetDefaultParameterMap('affine')
        affineParameterMap['MaximumNumberOfSamplingAttempts'] = '0'
        affineParameterMap['Metric'] = ['AdvancedMeanSquares']
        affineParameterMap['MaximumNumberOfIterations'] = ['{}'.format(iterations)]
        affineParameterMap['Optimizer'] = ['StandardGradientDescent']
        affineParameterMap['NumberOfResolutions'] = '{}'.format(resolutions)  
        
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixedImage)
        elastixImageFilter.SetMovingImage(movingImage)
        elastixImageFilter.SetParameterMap(affineParameterMap)
        elastixImageFilter.Execute()
        self.atlas_affine = elastixImageFilter.GetResultImage()
        transformParameterMap = elastixImageFilter.GetTransformParameterMap()[0] 
        self.affine = [float(i) for i in transformParameterMap['TransformParameters']]
        return self.atlas_affine
        
    def register_lddmm(self, affine_img, target_img, alphaList=[0.05], scaleList=[0.0625, 0.125, 0.25],
            epsilonList=1e-7, sigma=None, useMI=False, iterations=200, verbose=True):
        if sigma == None:
            sigma = (0.1/target_img.GetNumberOfPixels())
        # TODO: Add sigma param in ndreg and recompile
        (field, invField) = ndreg.imgMetamorphosisComposite(affine_img, target_img, alphaList=alphaList,
                                              scaleList = scaleList, epsilonList=epsilonList,
                                              useMI=useMI, iterations=iterations, verbose=verbose)
        affineField = ndreg.affineToField(self.affine, field.GetSize(), field.GetSpacing())
        self.fieldComposite = ndreg.fieldApplyField(field, affineField)

        invAffineField = ndreg.affineToField(affineInverse(self.affine), invField.GetSize(), invField.GetSpacing())
        self.invFieldComposite = ndreg.fieldApplyField(invAffineField, invField)
       
        refImg_lddmm = ndreg.imgApplyField(affine_img, self.fieldComposite, size=inImgReoriented.GetSize(), spacing=inImgReoriented.GetSpacing())
        