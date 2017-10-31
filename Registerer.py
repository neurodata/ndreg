import ndreg
import csv
import SimpleITK as sitk


class Registerer:

    def __init__(self, img, atlas, inOrient, refOrient):
        if type(img) != sitk.SimpleITK.Image:
            raise Exception("Please convert your image into a SimpleITK Image")
        if type(atlas) != sitk.SimpleITK.Image:
            raise Exception("Please convert your image into a SimpleITK Image")
        else:
            self.img = img
            self.atlas = atlas
            self.affine = []
            self.inOrient = inOrient
            self.refOrient = refOrient
            self.atlas_affine = None
            self.fieldComposite = None
            self.invFieldComposite = None

    def register_affine(self, spacing, iterations=2000.0, resolutions=8.0):
        self.img_reoriented = ndreg.imgReorient(self.img, self.inOrient, self.refOrient)
        atlas_ds = sitk.Clamp(ndreg.imgResample(self.atlas, spacing), upperBound=ndreg.imgPercentile(self.atlas, 0.99))
        img_ds = sitk.Clamp(ndreg.imgResample(self.img_reoriented, spacing), upperBound=ndreg.imgPercentile(self.img_reoriented, 0.99))
        # normalize
        max_val = ndreg.imgPercentile(img_ds, 0.999)
        min_val = ndreg.imgPercentile(img_ds, 0.001)
        self.img_ds = (img_ds - min_val)/(max_val - min_val)
        
        max_val = ndreg.imgPercentile(atlas_ds, 0.999)
        min_val = ndreg.imgPercentile(atlas_ds, 0.001)
        self.atlas_ds = (atlas_ds - min_val)/(max_val - min_val)
        
        fixedImage = self.img_ds
        movingImage = self.atlas_ds
        
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
        
    def register_lddmm(self, affine_img=None, target_img=None, alphaList=[0.05], scaleList=[0.0625, 0.125, 0.25],
            epsilonList=1e-7, sigma=None, useMI=False, iterations=200, verbose=True):
        if affine_img == None and self.atlas_affine is None:
           raise Exception("Perform the affine registration first")
        elif affine_img == None:
            affine_img = self.atlas_affine
        if target_img == None:
            target_img = self.img_ds
        if sigma == None:
            sigma = (0.1/target_img.GetNumberOfPixels())
        # TODO: Add sigma param in ndreg and recompile
        (self.field, self.invField) = ndreg.imgMetamorphosisComposite(affine_img, target_img, alphaList=alphaList,
                                              scaleList = scaleList, epsilonList=epsilonList,
                                              useMI=useMI, iterations=iterations, verbose=verbose)
        affineField = ndreg.affineToField(self.affine, self.field.GetSize(), self.field.GetSpacing())
        self.fieldComposite = ndreg.fieldApplyField(self.field, affineField)

        invAffineField = ndreg.affineToField(ndreg.affineInverse(self.affine), self.invField.GetSize(), self.invField.GetSpacing())
        self.invFieldComposite = ndreg.fieldApplyField(invAffineField, self.invField)
       
        self.ref_lddmm = ndreg.imgApplyField(affine_img, self.fieldComposite, size=self.img_reoriented.GetSize(), spacing=self.img_reoriented.GetSpacing())
        return self.ref_lddmm
        
    def evaluate_registration(self, source_fiducial_file, target_fiducial_file):
        reader = csv.reader(file(source_fiducial_file))
        # skip first line, dont need it
        reader.next()
        # get the coordinate system
        coordinateSystem = reader.next()[0].split()[3]
        columns = reader.next()[1:]
        atlasLabels = []
        atlasLocs = []
        # atlas origin is the geometric mean
        atlasOrigin = np.array(refImg.GetSize())/2.0
        for i in reader:
            atlasLabels.append([i[-3]])
            atlasLocs.append([float(j) for j in i[1:4]])