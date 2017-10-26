import ndreg
import SimpleITK as sitk


class Registerer:

    def ___init__(self, img, atlas):
        if type(img) != SimpleITK.SimpleITK.Image:
            raise Exception("Please convert your image into a SimpleITK Image")
        else if type(atlas) != SimpleITK.SimpleITK.Image:
            raise Exception("Please convert your image into a SimpleITK Image")
        else:
            self.img = img
            self.atlas = atlas
            self.affine = []
            self.atlas_affine = None
            self.fieldComposite = None
            self.invFieldComposite = None

    def register_affine(self):
        fixedImage = self.img
        movingImage = self.atlas
       	# set parameters
        affineParameterMap = sitk.GetDefaultParameterMap('affine')
        affineParameterMap['Metric'] = ['AdvancedMeanSquares']
        affineParameterMap['MaximumNumberOfIterations'] = ['2000.00']
        affineParameterMap['Optimizer'] = ['StandardGradientDescent']
        affineParameterMap['NumberOfResolutions'] = '8.00'  
        
	elastixImageFilter = sitk.ElastixImageFilter()
	elastixImageFilter.SetFixedImage(fixedImage)
	elastixImageFilter.SetMovingImage(movingImage)
	elastixImageFilter.SetParameterMap(affineParameterMap)
	elastixImageFilter.Execute()
		
	self.atlas_affine = elastixImageFilter.GetResultImage()
	transformParameterMap = elastixImageFilter.GetTransformParameterMap()[0]	
	
	self.affine = [float(i) for i in transformParameterMap['TransformParameters']]
	
	    
    def register_lddmm(self, affine_img, target_img, alphaList=[0.05], scaleList=[0.0625, 0.125, 0.25],
            epsilonList=1e-7, sigma=None, useMI=False, iterations=200, verbose=True):
        if sigma == None:
            sigma = (0.1/inImg_ds.GetNumberOfPixels())
        (field, invField) = ndreg.imgMetamorphosisComposite(affine_img, target_img, alphaList=alphaList,
                                              scaleList = scaleList, epsilonList=epsilonList, sigma=sigma,
                                              useMI=useMI, iterations=iterations, verbose=verbose)
        affineField = ndreg.affineToField(self.affine, field.GetSize(), field.GetSpacing())
        self.fieldComposite = ndreg.fieldApplyField(field, affineField)

        invAffineField = ndreg.affineToField(affineInverse(self.affine), invField.GetSize(), invField.GetSpacing())
        self.invFieldComposite = ndreg.fieldApplyField(invAffineField, invField)
       
        refImg_lddmm = ndreg.imgApplyField(affine_img, self.fieldComposite, size=inImgReoriented.GetSize(), spacing=inImgReoriented.GetSpacing())
