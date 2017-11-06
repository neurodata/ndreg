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
            self.affine_s2t = None
            self.fieldComposite = None
            self.invFieldComposite = None
            self.source_lddmm = None

    def register_affine(self, spacing, iterations=2000.0, resolutions=8.0):
        # reorient the target to match the source
        self.target_reoriented = ndreg.imgReorient(self.target, self.targetOrient, self.sourceOrient)
        source_ds = sitk.Clamp(ndreg.imgResample(self.source, [spacing]*3), upperBound=ndreg.imgPercentile(self.source, 0.99))
        target_ds = sitk.Clamp(ndreg.imgResample(self.target_reoriented, [spacing]*3), upperBound=ndreg.imgPercentile(self.target_reoriented, 0.99))
        # normalize
        max_val = ndreg.imgPercentile(target_ds, 0.999)
        min_val = ndreg.imgPercentile(target_ds, 0.001)
        self.target_ds = (target_ds - min_val)/(max_val - min_val)
        
        max_val = ndreg.imgPercentile(source_ds, 0.999)
        min_val = ndreg.imgPercentile(source_ds, 0.001)
        self.source_ds = (source_ds - min_val)/(max_val - min_val)
        
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
        # get the affine transformed source image
        self.affine_s2t = elastixImageFilter.GetResultImage()
        transformParameterMap = elastixImageFilter.GetTransformParameterMap()[0] 
        # save the affine transformation
        self.affine = [float(i) for i in transformParameterMap['TransformParameters']]
        return self.affine_s2t
        
    def register_lddmm(self, affine_img=None, target_img=None, alphaList=[0.05], scaleList=[0.0625, 0.125, 0.25],
            epsilonList=1e-7, sigma=None, useMI=False, iterations=200, verbose=True):
        if affine_img == None and self.affine_s2t is None:
           raise Exception("Perform the affine registration first")
        elif affine_img == None:
            affine_img = self.affine_s2t
        if target_img == None:
            target_img = self.target_ds
        if sigma == None:
            sigma = (0.1/target_img.GetNumberOfPixels())
        # TODO: Add sigma param in ndreg and recompile
        (self.field, self.invField) = ndreg.imgMetamorphosisComposite(affine_img, target_img, alphaList=alphaList,
                                              scaleList = scaleList, epsilonList=epsilonList,
                                              useMI=useMI, iterations=iterations, verbose=verbose)
        affineField = ndreg.affineToField(self.affine, self.field.GetSize(), self.field.GetSpacing())
        self.fieldComposite = ndreg.fieldApplyField(self.field, affineField)

        invAffineField = ndreg.affineToField(ndreg.affineInverse(self.affine), self.invField.GetSize(),
                                             self.invField.GetSpacing())
        self.invFieldComposite = ndreg.fieldApplyField(invAffineField, self.invField)
       
        self.source_lddmm = ndreg.imgApplyField(affine_img, self.fieldComposite, size=self.target_reoriented.GetSize(),
                                             spacing=self.target_reoriented.GetSpacing())
        return self.source_lddmm
        
    def checkerboard_image(self):
        if self.affine_s2t is None:
            raise Exception("Perform a registration first. This can be just affine or affine + LDDMM")
        elif self.source_lddmm is None:
            ndreg.imgShow(ndreg.imgChecker(self.target_ds, self.affine_s2t))
        else:
            ndreg.imgShow(ndreg.imgChecker(self.target_ds, self.source_lddmm))
            
    def evaluate_affine_registration(self, source_fiducial_file, target_fiducial_file, scale_source, scale_target, orientation_source, orientation_t_initial, orientation_t_final):
        # load the landmarks for each brain
        landmarks_source = self._parse_fiducial_file(source_fiducial_file, scale_source)
        landmarks_target = self._parse_fiducial_file(target_fiducial_file, scale_target)
        # reorient the source fiducials so they match the orientation of
        # target image. then we can apply our transformations
        landmarks_source_reoriented = self._reorient_landmarks(landmarks_source, orientation_source, 
                                                               orientation_t_initial, self.source)
        landmarks_source_affine = self._apply_affine(landmarks_source_reoriented, self.affine)
        landmarks_source_affine_reoriented = self._reorient_landmarks(landmarks_source_reoriented,
                                                                     orientation_t_initial, orientation_t_final, 
                                                                      self.source)
        mse = self._compute_error(landmarks_source_affine_reoriented, landmarks_target)
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
#         print(locs)
#         print(swap)
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
#             landmarks.append([i[-3], abs(float(i[1])*scale), abs(float(i[2])*scale), abs(float(i[3])*scale)])
            landmarks.append([abs(float(i[1])*scale), abs(float(i[2])*scale), abs(float(i[3])*scale)])
        return np.array(landmarks)
    
    def _get_landmark_pts(self, landmarks):
        return np.array([[float(i[1]), float(i[2]), float(i[3])] for i in landmarks.GetLandmarks()])

    def _apply_affine(self, landmarks, affine):
        landmarks_affine = []
        affine_mat = np.matrix(np.reshape(self.affine[:9], (3,3)))
        affine_t = np.array(self.affine[9:])
        for i in np.array(landmarks):
            landmarks_affine.append(((affine_mat * np.matrix(i).T).T + affine_t).A)
        return np.array(landmarks_affine)
            