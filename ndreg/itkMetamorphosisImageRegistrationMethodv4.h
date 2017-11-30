#ifndef __itkMetamorphosisImageRegistrationMethodv4_h
#define __itkMetamorphosisImageRegistrationMethodv4_h

#include "itkTimeVaryingVelocityFieldImageRegistrationMethodv4.h"
#include "itkTimeVaryingVelocityFieldSemiLagrangianTransform.h"
#include "itkForwardFFTImageFilter.h"
#include "itkInverseFFTImageFilter.h"
#include "itkFFTPadImageFilter.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkImportImageFilter.h"
#include "itkComposeImageFilter.h"
#include "itkVectorIndexSelectionCastImageFilter.h"
#include "itkVectorMagnitudeImageFilter.h"
#include "itkGradientImageFilter.h"
#include "itkDisplacementFieldJacobianDeterminantFilter.h"
#include "itkStatisticsImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkJoinSeriesImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkWrapExtrapolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkImageMaskSpatialObject.h"
#include "itkSpatialObjectToImageFilter.h"

namespace itk
{
template<class TInputImage, class TOutputImage>
class ConstantImageFilter:
public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ConstantImageFilter                             Self;
  typedef ImageToImageFilter< TInputImage, TOutputImage > Superclass;
  typedef SmartPointer< Self >                            Pointer;
 
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(ConstantImageFilter, ImageToImageFilter);

  typedef TOutputImage OutputImageType;
  typedef typename OutputImageType::PixelType       OutputPixelType;
  typedef ImageRegionIterator<OutputImageType> OutputImageIteratorType;

  itkSetMacro(Constant, OutputPixelType);
  itkGetConstMacro(Constant, OutputPixelType);

 
protected:
  ConstantImageFilter(){}
  ~ConstantImageFilter(){}
 
  /** Does the real work. */
  void ThreadedGenerateData(  const typename OutputImageType::RegionType& outputRegionForThread, ThreadIdType threadId)
  {
    ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());
    OutputPixelType constant = this->GetConstant();
    OutputImageIteratorType  it(this->GetOutput(),outputRegionForThread);

    while(!it.IsAtEnd())
    {
      it.Set(constant);
      ++it;
      progress.CompletedPixel();
    }
  }
 
private:
  ConstantImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented
 
  OutputPixelType m_Constant;
};

/** \class MetamorphosisImageRegistrationMethodv4
* \breif Perfoms metamorphosis registration between images
*
* \author Kwane Kutten
*
* \ingroup ITKRegistrationMethodsv4
*/

template<  typename TFixedImage,
           typename TMovingImage = TFixedImage >
class MetamorphosisImageRegistrationMethodv4:
public TimeVaryingVelocityFieldImageRegistrationMethodv4<TFixedImage, TMovingImage, TimeVaryingVelocityFieldSemiLagrangianTransform<double, TFixedImage::ImageDimension> >
{
public:
  /** Standard class typedefs. */
  typedef MetamorphosisImageRegistrationMethodv4                  Self;
  typedef TimeVaryingVelocityFieldImageRegistrationMethodv4<TFixedImage, TMovingImage, TimeVaryingVelocityFieldSemiLagrangianTransform<double, TFixedImage::ImageDimension> > Superclass;
  typedef SmartPointer<Self>                                      Pointer;
  typedef SmartPointer<const Self>                                ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information */
  itkTypeMacro(MetamorphosisImageRegistrationMethodv4, TimeVaryingVelocityFieldImageRegistrationMethodv4);

  /** Concept checking */
#ifdef ITK_USE_CONCEPT_CHECKING
  itkConceptMacro(MovingSameDimensionCheck, (Concept::SameDimension<TFixedImage::ImageDimension,TMovingImage::ImageDimension>));
#endif

  /** Image typedefs */
  itkStaticConstMacro(ImageDimension, unsigned int, TFixedImage::ImageDimension);

  typedef TFixedImage                               FixedImageType;
  typedef typename FixedImageType::Pointer          FixedImagePointer;

  typedef TMovingImage                              MovingImageType;
  typedef typename MovingImageType::Pointer         MovingImagePointer;

  typedef typename Superclass::VirtualImageType     VirtualImageType;
  typedef typename VirtualImageType::Pointer        VirtualImagePointer;
  typedef typename VirtualImageType::PixelType      VirtualPixelType;

  typedef Image<unsigned char, ImageDimension>      MaskImageType;
  typedef typename MaskImageType::Pointer           MaskImagePointer;
  typedef ImageMaskSpatialObject<ImageDimension>        MaskType;  
  typedef typename MaskType::Pointer                MaskPointer;
  typedef typename MaskType::ConstPointer           MaskConstPointer;

  typedef VirtualImageType                          BiasImageType;
  typedef typename BiasImageType::Pointer           BiasImagePointer;

  typedef typename Superclass::OutputTransformType  OutputTransformType;
  typedef typename OutputTransformType::ScalarType  RealType;

  typedef typename OutputTransformType::DisplacementFieldType          FieldType;
  typedef typename FieldType::Pointer                                  FieldPointer;
  typedef typename FieldType::PixelType                                VectorType;

  typedef Image<typename VirtualImageType::PixelType,ImageDimension+1> TimeVaryingImageType;
  typedef typename TimeVaryingImageType::Pointer                       TimeVaryingImagePointer;

  typedef typename OutputTransformType::TimeVaryingVelocityFieldType   TimeVaryingFieldType;
  typedef typename TimeVaryingFieldType::Pointer                       TimeVaryingFieldPointer;

  // Metric typedefs
  typedef typename Superclass::ImageMetricType     ImageMetricType;
  typedef typename ImageMetricType::Pointer        ImageMetricPointer;
  typedef typename ImageMetricType::MeasureType    MetricValueType;
  typedef typename ImageMetricType::DerivativeType MetricDerivativeType;
  typedef typename ImageMetricType::MetricTraits   MetricTraits;

  // Filter typedefs
  typedef typename MetricTraits::FixedImageGradientFilterType  FixedImageGradientFilterType;
  typedef typename FixedImageGradientFilterType::Pointer       FixedImageGradientFilterPointer;
  typedef GradientImageFilter<FixedImageType, double, double>  DefaultFixedImageGradientFilterType;
  typedef ConstantImageFilter<VirtualImageType, typename ImageMetricType::FixedImageGradientImageType> FixedImageConstantGradientFilterType;

  typedef typename MetricTraits::MovingImageGradientFilterType MovingImageGradientFilterType;
  typedef typename MovingImageGradientFilterType::Pointer      MovingImageGradientFilterPointer;
  typedef GradientImageFilter<MovingImageType, double, double> DefaultMovingImageGradientFilterType;
  typedef ConstantImageFilter<VirtualImageType, typename ImageMetricType::MovingImageGradientImageType> MovingImageConstantGradientFilterType;

  /** Public member functions */
  itkSetMacro(Scale, double);
  itkGetConstMacro(Scale, double);
  itkSetMacro(RegistrationSmoothness, double);
  itkGetConstMacro(RegistrationSmoothness,double);
  itkSetMacro(BiasSmoothness, double);
  itkGetConstMacro(BiasSmoothness,double);
  itkSetMacro(Sigma, double);
  itkGetConstMacro(Sigma, double);
  itkSetMacro(Mu, double);
  itkGetConstMacro(Mu, double);
  itkSetMacro(Gamma, double);
  itkGetConstMacro(Gamma, double);
  itkSetMacro(MinLearningRate, double)
  itkGetConstMacro(MinLearningRate, double);
  itkSetMacro(MinImageEnergyFraction, double);
  itkGetConstMacro(MinImageEnergyFraction, double);
  itkSetMacro(NumberOfTimeSteps, unsigned int);
  itkGetConstMacro(NumberOfTimeSteps, unsigned int);
  itkSetMacro(NumberOfIterations, unsigned int);
  itkGetConstMacro(NumberOfIterations, unsigned int);
  itkBooleanMacro(UseJacobian);
  itkSetMacro(UseJacobian, bool);
  itkGetConstMacro(UseJacobian, bool);
  itkBooleanMacro(UseBias);
  itkSetMacro(UseBias, bool);
  itkGetConstMacro(UseBias, bool);

  double GetVelocityEnergy();
  double GetRateEnergy();
  double GetImageEnergy(VirtualImagePointer movingImage, MaskPointer movingMask=ITK_NULLPTR);
  double GetImageEnergy();
  double GetImageEnergyFraction();
  double GetEnergy();
  double GetLength();
  BiasImagePointer GetBias();

protected:
  MetamorphosisImageRegistrationMethodv4();
  ~MetamorphosisImageRegistrationMethodv4(){};
  TimeVaryingImagePointer ApplyKernel(TimeVaryingImagePointer kernel, TimeVaryingImagePointer image);
  TimeVaryingFieldPointer ApplyKernel(TimeVaryingImagePointer kernel, TimeVaryingFieldPointer image);
  double CalculateNorm(TimeVaryingImagePointer image);
  double CalculateNorm(TimeVaryingFieldPointer field);
  void InitializeKernels(TimeVaryingImagePointer kernel, TimeVaryingImagePointer inverseKernel, double alpha, double gamma);
  void Initialize();
  void IntegrateRate();
  FieldPointer GetMetricDerivative(FieldPointer field, bool useImageGradients);
  void UpdateControls(); 
  void StartOptimization();
  void GenerateData();
  void PrintSelf(std::ostream& os, Indent indent) const;

private:
  MetamorphosisImageRegistrationMethodv4(const Self&);  // Intentionally not implemened
  void operator=(const Self&);    //Intentionally not implemented

  double m_Scale;
  double m_RegistrationSmoothness;
  double m_BiasSmoothness;
  double m_Sigma;
  double m_Mu;
  double m_Gamma;
  double m_MinLearningRate;
  double m_MinImageEnergyFraction;
  double m_MaxImageEnergy;
  double m_MinImageEnergy;
  unsigned int m_NumberOfTimeSteps;
  unsigned int m_NumberOfIterations;
  bool m_UseJacobian;
  bool m_UseBias;
  double m_TimeStep;
  double m_VoxelVolume;
  double m_Energy;
  bool m_RecalculateEnergy;
  bool m_IsConverged;
  VirtualImagePointer m_VirtualImage;
  VirtualImagePointer m_ForwardImage;
  MaskImagePointer    m_MovingMaskImage;
  MaskImagePointer    m_ForwardMaskImage;
  typename VirtualImageType::PointType m_CenterPoint;
  TimeVaryingImagePointer m_VelocityKernel;
  TimeVaryingImagePointer m_InverseVelocityKernel;
  TimeVaryingImagePointer m_RateKernel;
  TimeVaryingImagePointer m_InverseRateKernel;
  TimeVaryingImagePointer m_Rate;
  VirtualImagePointer m_Bias;

  typename MovingImageConstantGradientFilterType::Pointer m_MovingImageConstantGradientFilter;
  typename FixedImageConstantGradientFilterType::Pointer  m_FixedImageConstantGradientFilter;

}; // End class MetamorphosisImageRegistrationMethodv4


} // End namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMetamorphosisImageRegistrationMethodv4.hxx"
#endif

#endif
