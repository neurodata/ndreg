import ndreg
import csv
import numpy as np
import SimpleITK as sitk
# from IPython.core.debugger import Tracer


class Registerer:

    def __init__(self, source, target, sourceOrient, targetOrient):
        if type(source) != sitk.SimpleITK.Image:
            raise Exception("Please convert your image into a SimpleITK Image")
        if type(target) != sitk.SimpleITK.Image:
            raise Exception("Please convert your image into a SimpleITK Image")
        else:
            self.source = source
            self.target = target
            self.affine = []
            self.targetOrient = targetOrient
            self.sourceOrient = sourceOrient
            self.source_affine = None
            self.field = None
            self.invField = None
            self.fieldComposite = None
            self.invFieldComposite = None
            self.source_lddmm = None

    def register_affine(self, spacing, iterations=2000.0, resolutions=8.0):
        # reorient the target to match the source
        self.target_reoriented = ndreg.imgReorient(self.target, self.targetOrient, self.sourceOrient)
        source_ds = sitk.Clamp(ndreg.imgResample(self.source, [spacing]*3), upperBound=ndreg.imgPercentile(self.source, 0.99))
        target_ds = sitk.Clamp(ndreg.imgResample(self.target_reoriented, [spacing]*3), upperBound=ndreg.imgPercentile(self.target_reoriented, 0.99))
        # normalize
        self.target_ds = self._normalize_image(target_ds)
        self.source_ds = self._normalize_image(source_ds)
        
        movingImage = self.source_ds
        fixedImage = self.target_ds
        
        # set parameters
        affineParameterMap = sitk.GetDefaultParameterMap('affine')
        affineParameterMap['MaximumNumberOfSamplingAttempts'] = '0'
        affineParameterMap['Metric'] = ['AdvancedMeanSquares']
        affineParameterMap['MaximumNumberOfIterations'] = ['{}'.format(iterations)]
        affineParameterMap['Optimizer'] = ['StandardGradientDescent']
        affineParameterMap['NumberOfResolutions'] = '{}'.format(resolutions)  
        
        # initialize registration object
        elastixImageFilter = sitk.ElastixImageFilter()
        # set source and target images
        elastixImageFilter.SetFixedImage(fixedImage)
        elastixImageFilter.SetMovingImage(movingImage)
        # set parameter map
        elastixImageFilter.SetParameterMap(affineParameterMap)
        # run the registration
        elastixImageFilter.Execute()
        self.elastix_img_filt = elastixImageFilter
        # get the affine transformed source image
        self.source_affine = elastixImageFilter.GetResultImage()
        transformParameterMap = elastixImageFilter.GetTransformParameterMap()[0] 
        # save the affine transformation
        self.affine = [float(i) for i in transformParameterMap['TransformParameters']]
        return self.source_affine
        
    def register_lddmm(self, affine_img=None, target_img=None, alpha_list=0.05, scale_list=[0.0625, 0.125, 0.25], 
                       epsilon_list=1e-7, sigma=None, use_mi=False, iterations=200, verbose=True, out_dir=''):
        if affine_img == None and self.source_affine is None:
           raise Exception("Perform the affine registration first")
        elif affine_img == None:
            # normalize affine image [0, 1]
            affine_img = self._normalize_image(self.source_affine)
        if target_img == None:
            target_img = self.target_ds
        if sigma == None:
            sigma = (0.1/target_img.GetNumberOfPixels())
        
        # TODO: Add sigma param in ndreg and recompile
        (self.field, self.invField) = ndreg.imgMetamorphosisComposite(affine_img, target_img,
                                                                      alphaList=alpha_list,
                                                                      scaleList=scale_list,
                                                                      epsilonList=epsilon_list,
                                                                      sigma=sigma,
                                                                      useMI=use_mi, 
                                                                      iterations=iterations, 
                                                                      verbose=verbose,
                                                                      outDirPath=out_dir)
        affineField = ndreg.affineToField(self.affine, self.field.GetSize(), self.field.GetSpacing())
        self.fieldComposite = ndreg.fieldApplyField(self.field, affineField)

        invAffineField = ndreg.affineToField(ndreg.affineInverse(self.affine), self.invField.GetSize(),
                                             self.invField.GetSpacing())
        self.invFieldComposite = ndreg.fieldApplyField(invAffineField, self.invField)
       
        self.source_lddmm = ndreg.imgApplyField(self.source, self.fieldComposite, 
                                                size=self.target_reoriented.GetSize(), 
                                                spacing=self.target_reoriented.GetSpacing())
        return self.source_lddmm
        
    def checkerboard_image(self, vmax=None):
        if vmax == None:
            vmax = ndreg.imgPercentile(self.target_ds, 0.99)
        if self.source_affine is None:
            raise Exception("Perform a registration first. This can be just affine or affine + LDDMM")
        elif self.source_lddmm is None:
            ndreg.imgShow(ndreg.imgChecker(self._normalize_image(self.target_ds), 
                                           self._normalize_image(self.source_affine)),
                          vmax=1.0)
        else:
            ndreg.imgShow(ndreg.imgChecker(self._normalize_image(self.target_ds), 
                                           self._normalize_image(self.source_lddmm)), 
                          vmax=1.0)
            
    def evaluate_affine_registration(self, source_fiducial_file, target_fiducial_file, scale_source, scale_target,
                                     orientation_source_fid, orientation_target_fid):
        # load the landmarks for each brain
        landmarks_source = self._parse_fiducial_file(source_fiducial_file, scale_source)
        landmarks_target = self._parse_fiducial_file(target_fiducial_file, scale_target)
        # reorient the source fiducials so they match the orientation of
        # source we calculated transformation on. then we can apply our transformations.
        landmarks_source_r = self._reorient_landmarks(landmarks_source, orientation_source_fid, 
                                                               self.sourceOrient, self.source)
        landmarks_source_a = self._apply_affine(landmarks_source_r)
        landmarks_source_ar = self._reorient_landmarks(landmarks_source_a, self.sourceOrient, 
                                                       orientation_target_fid, self.target)
        mse = self._compute_error(landmarks_source_ar, landmarks_target)
        return mse
    
    def evaluate_lddmm_registration(self, source_fiducial_file, target_fiducial_file, scale_source, scale_target,
                                   orientation_source_fid, orientation_target_fid):
        if self.fieldComposite is None:
            raise Exception("Perform LDDMM registration first")
        landmarks_source = self._parse_fiducial_file(source_fiducial_file, scale_source)
        landmarks_target = self._parse_fiducial_file(target_fiducial_file, scale_target)
        landmarks_source_r = self._reorient_landmarks(landmarks_source, orientation_source_fid, 
                                                      self.sourceOrient, self.source)
        landmarks_source_a = self._apply_affine(landmarks_source_r)
        landmarks_source_lddmm = self._lmk_apply_field(landmarks_source_a, self.field)
        landmarks_source_lddmm_r = self._reorient_landmarks(landmarks_source_lddmm, self.sourceOrient,
                                                            orientation_target_fid, self.target)
        mse = self._compute_error(np.array(landmarks_source_lddmm_r), np.array(landmarks_target))
        return mse
    
    def _reorient_landmarks(self, landmarks, in_orient, out_orient, in_img):
        orient = {'l': 'lr', 'a': 'ap', 's': 'si','r': 'rl', 'p':'pa', 'i': 'is'}
        order_in = []
        order_out = []
        for i in range(len(in_orient)):
            order_in.append(orient['{}'.format(in_orient[i].lower())])
            order_out.append(orient['{}'.format(out_orient[i].lower())])
        locs = []
        swap = []
        for i in range(len(order_in)):
            try:
                locs.append(order_in.index(order_out[i]))
                swap.append(0)
            except Exception as e:
                locs.append(order_in.index(order_out[i][::-1]))
                swap.append(1)
         # TODO: implement swap code
        for i in range(len(swap)):
            if swap[i]:
#                 print('swapping axis {}'.format(locs[i]))
                landmarks = self._flip_fiducials_along_axis(landmarks, in_img, axis=locs[i])
                
        landmarks_reoriented = np.array(landmarks)[:,locs]
        return landmarks_reoriented
    
    def _flip_fiducials_along_axis(self, fiducials, img, axis=2):
        if type(fiducials) == list: fiducials_new = np.array(fiducials).copy()
        else: fiducials_new = fiducials.copy() 
        offset = img.GetSize()[axis] * img.GetSpacing()[axis]
        fiducials_new[:,axis] = np.abs(fiducials_new[:,axis] - offset)
        return fiducials_new
        
    def _compute_error(self, ref_landmarks, img_landmarks):
        if ref_landmarks.shape[0] != img_landmarks.shape[0]:
            raise Exception("pass in the same number of fiducials")
        distances = np.zeros([len(ref_landmarks), 1])
        for i in range(len(ref_landmarks)):
#             print(ref_landmarks[i,:] - img_landmarks[i,:])
            distances[i] = np.linalg.norm(ref_landmarks[i,:] - img_landmarks[i,:])
        return distances
        
    def _parse_fiducial_file(self, f, scale):
        reader = csv.reader(file(f))
        # skip first line, dont need it
        reader.next()
        # get the coordinate system
        coordinateSystem = reader.next()[0].split()[3]
        columns = reader.next()[1:]
        landmarks = []
        for i in reader:
            landmarks.append(np.abs(np.array(i[1:4]).astype(np.float) * scale))
        return np.array(landmarks)
    
    def _lmk_apply_field(self, fids, field):
        dim = 3
        # Create transform
        field_copy = sitk.GetImageFromArray(sitk.GetArrayFromImage(field))
        field_copy.CopyInformation(field)
        transform = sitk.DisplacementFieldTransform(dim)
        transform.SetInterpolator(sitk.sitkLinear)
        transform.SetDisplacementField(sitk.Cast(field_copy, sitk.sitkVectorFloat64))
        return np.array([transform.TransformPoint(i) for i in fids])
    
    def _apply_affine(self, fids):
        p = self.elastix_img_filt.GetTransformParameterMap()[0]
        center = p['CenterOfRotationPoint']
        center_f = np.array([float(i) for i in center])
        at = sitk.AffineTransform(3)
        at.SetCenter(center_f)
        at.SetMatrix(self.affine[:9])
        at.SetTranslation(self.affine[9:])
        return np.array([at.TransformPoint(i) for i in fids])
        
        
    def _normalize_image(self, img, low_bound=None, up_bound=0.999):
        min_val = 0.0
        if low_bound is not None:
            min_val = ndreg.imgPercentile(img, low_bound)
        max_val = ndreg.imgPercentile(img, up_bound)
        
        return (img - min_val)/(max_val - min_val)