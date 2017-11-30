#ifndef __itkMetamorphosisImageRegistrationMethodv4_hxx
#define __itkMetamorphosisImageRegistrationMethodv4_hxx
#include "itkMetamorphosisImageRegistrationMethodv4.h"

namespace itk
{

template<typename TFixedImage, typename TMovingImage>
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
MetamorphosisImageRegistrationMethodv4()
{
  m_Scale = 1.0;                      // 1
  m_RegistrationSmoothness = 0.01;    // 0.01
  m_BiasSmoothness = 0.05;            // 0.05
  m_Mu = 0.1;                         // 0.1
  m_Sigma = 1.0;                      // 1
  m_Gamma = 1.0;                      // 1
  this->SetLearningRate(1e-3);        // 1e-3
  this->SetMinLearningRate(1e-10);     // 1e-8
  m_MinImageEnergyFraction = 0;  
  m_MinImageEnergy = 0;
  m_MaxImageEnergy = 0;
  m_NumberOfTimeSteps = 10;           // 4 
  m_NumberOfIterations = 100;         // 20
  m_UseJacobian = true;
  m_UseBias = true;
  m_RecalculateEnergy = true;
  this->m_CurrentIteration = 0;
  this->m_IsConverged = false;

  m_VelocityKernel = TimeVaryingImageType::New();                // K_V
  m_InverseVelocityKernel = TimeVaryingImageType::New();         // L_V
  m_RateKernel = TimeVaryingImageType::New();                    // K_R
  m_InverseRateKernel = TimeVaryingImageType::New();             // L_R
  m_Rate = TimeVaryingImageType::New();                          // r
  m_Bias = VirtualImageType::New();                              // B
  m_VirtualImage = VirtualImageType::New();

  typedef typename ImageMetricType::FixedImageGradientImageType::PixelType             FixedGradientPixelType;
  m_FixedImageConstantGradientFilter = FixedImageConstantGradientFilterType::New();
  m_FixedImageConstantGradientFilter->SetConstant(NumericTraits<FixedGradientPixelType>::One);

  typedef typename ImageMetricType::MovingImageGradientImageType::PixelType             MovingGradientPixelType;
  m_MovingImageConstantGradientFilter = MovingImageConstantGradientFilterType::New();
  m_MovingImageConstantGradientFilter->SetConstant(NumericTraits<MovingGradientPixelType>::One);

  this->SetMetric(MeanSquaresImageToImageMetricv4<FixedImageType, MovingImageType>::New());
}

template<typename TFixedImage, typename TMovingImage>
typename MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::TimeVaryingImagePointer
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
ApplyKernel(TimeVaryingImagePointer kernel, TimeVaryingImagePointer image)
{
  // Calculate the Fourier transform of image.
  typedef ForwardFFTImageFilter<TimeVaryingImageType>    FFTType;
  typename FFTType::Pointer                     fft = FFTType::New();
  fft->SetInput(image);

  //...multiply it by the kernel...
  typedef typename FFTType::OutputImageType  ComplexImageType;
  typedef MultiplyImageFilter<ComplexImageType,TimeVaryingImageType,ComplexImageType>      ComplexImageMultiplierType;
  typename ComplexImageMultiplierType::Pointer  multiplier = ComplexImageMultiplierType::New();
  multiplier->SetInput1(fft->GetOutput());		// Fourier-Transform of image
  multiplier->SetInput2(kernel);			// Kernel

  // ...and finaly take the inverse Fourier transform.
  typedef InverseFFTImageFilter<ComplexImageType,TimeVaryingImageType>  IFFTType;
  typename IFFTType::Pointer                    ifft = IFFTType::New();
  ifft->SetInput(multiplier->GetOutput());
  ifft->Update();

  return ifft->GetOutput();
}

template<typename TFixedImage, typename TMovingImage>
typename MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::TimeVaryingFieldPointer
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
ApplyKernel(TimeVaryingImagePointer kernel, TimeVaryingFieldPointer field)
{
  // Apply kernel to each component of field
  typedef ComposeImageFilter<TimeVaryingImageType,TimeVaryingFieldType>   ComponentComposerType;
  typename ComponentComposerType::Pointer   componentComposer = ComponentComposerType::New();

  for(unsigned int i = 0; i < ImageDimension; i++)
  {
    typedef VectorIndexSelectionCastImageFilter<TimeVaryingFieldType,TimeVaryingImageType>    ComponentExtractorType;
    typename ComponentExtractorType::Pointer      componentExtractor = ComponentExtractorType::New();
    componentExtractor->SetInput(field);
    componentExtractor->SetIndex(i);
    componentExtractor->Update();

    componentComposer->SetInput(i,ApplyKernel(kernel,componentExtractor->GetOutput()));
  }
  componentComposer->Update();

  return componentComposer->GetOutput();
}

template<typename TFixedImage, typename TMovingImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
InitializeKernels(TimeVaryingImagePointer kernel, TimeVaryingImagePointer inverseKernel, double alpha, double gamma)
{
  typename TimeVaryingImageType::IndexType   index = this->m_OutputTransform->GetVelocityField()->GetLargestPossibleRegion().GetIndex();
  typename TimeVaryingImageType::SizeType    size = this->m_OutputTransform->GetVelocityField()->GetLargestPossibleRegion().GetSize();
  typename TimeVaryingImageType::SpacingType spacing = this->m_OutputTransform->GetVelocityField()->GetSpacing(); //
  typename TimeVaryingImageType::RegionType region(index,size);

  // Fill in kernels' values
  kernel->CopyInformation(this->m_OutputTransform->GetVelocityField());
  kernel->SetRegions(region);
  kernel->Allocate();

  inverseKernel->CopyInformation(this->m_OutputTransform->GetVelocityField());
  inverseKernel->SetRegions(region);
  inverseKernel->Allocate();

  typedef ImageRegionIteratorWithIndex<TimeVaryingImageType>         TimeVaryingImageIteratorType;
  TimeVaryingImageIteratorType KIt(kernel,kernel->GetLargestPossibleRegion());
  TimeVaryingImageIteratorType LIt(inverseKernel,inverseKernel->GetLargestPossibleRegion());

  for(KIt.GoToBegin(), LIt.GoToBegin(); !KIt.IsAtEnd(); ++KIt,++LIt)
  {
    typename TimeVaryingImageType::IndexType  k = KIt.GetIndex();	// Get the frequency index
    double  A, B;

    // For every dimension accumulate the sum in A
    unsigned int i;
    for(i = 0, A = gamma; i < ImageDimension; i++)
    {
      A += 2 * alpha * vcl_pow(size[i],2) * ( 1.0-cos(2*vnl_math::pi*k[i]/size[i]) );
      //A += 2 * alpha * vcl_pow(size[i]/m_Scale,2) * ( 1.0-cos(2*vnl_math::pi*k[i]/size[i]) );
      //A += 2 * alpha * vcl_pow(spacing[i],-2) * ( 1.0 - cos(2*vnl_math::pi*k[i]*spacing[i]) );
      //A += alpha * vcl_pow(2*vnl_math::pi*k[i]*spacing[i],2);
    }

    KIt.Set(vcl_pow(A,-2)); // Kernel
    LIt.Set(A);             // "Inverse" kernel
  }
}


template<typename TFixedImage, typename TMovingImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
Initialize()
{
  // Initialize velocity, v = 0 by seting velocity information based on fixed image and number of time steps
  typename FixedImageType::ConstPointer fixedImage = this->GetFixedImage();
  typename FixedImageType::RegionType fixedRegion = fixedImage->GetLargestPossibleRegion();

  typename TimeVaryingFieldType::IndexType velocityIndex; velocityIndex.Fill(0);
  typename TimeVaryingFieldType::SizeType  velocitySize; velocitySize.Fill(m_NumberOfTimeSteps);
  typename TimeVaryingFieldType::PointType velocityOrigin; velocityOrigin.Fill(0);
  typename TimeVaryingFieldType::DirectionType velocityDirection; velocityDirection.SetIdentity();
  typename TimeVaryingFieldType::SpacingType velocitySpacing; velocitySpacing.Fill(1);

  for(unsigned int i = 0; i < ImageDimension; i++)
  {
    velocityIndex[i] = fixedRegion.GetIndex()[i];
    velocitySize[i] =  vcl_floor(fixedRegion.GetSize()[i]*m_Scale + 1.0001);
    velocityOrigin[i] = fixedImage->GetOrigin()[i];
    velocitySpacing[i] = fixedImage->GetSpacing()[i] / m_Scale;
    for(unsigned int j = 0; j < ImageDimension; j++)
    {
      velocityDirection(i,j) = (fixedImage->GetDirection())(i,j);
    }
  }
  velocitySize[ImageDimension] = m_NumberOfTimeSteps;

  typename TimeVaryingFieldType::RegionType velocityRegion(velocityIndex, velocitySize);
  TimeVaryingFieldPointer velocity = TimeVaryingFieldType::New();
  velocity->SetOrigin(velocityOrigin);
  velocity->SetDirection(velocityDirection);
  velocity->SetSpacing(velocitySpacing);
  velocity->SetRegions(velocityRegion);
  velocity->Allocate();

  /*
  This filter uses FFT to smooth velocity fields.
  FFT runs most efficiently when each diminsion's size has a small prime factorization.
  Therefore we pad the velocity so that this condition is met.
  */
  typedef ForwardFFTImageFilter<TimeVaryingImageType> FFTType;
  unsigned int greatestPrimeFactor = FFTType::New()->GetSizeGreatestPrimeFactor();

  typedef FFTPadImageFilter<TimeVaryingFieldType> PadderType;
  typename PadderType::Pointer padder = PadderType::New();
  padder->SetSizeGreatestPrimeFactor(vnl_math_min((unsigned int)5, greatestPrimeFactor));
  padder->SetInput(velocity);
  padder->Update();
  velocity = padder->GetOutput();

  // Use size but not index from padder
  velocitySize = velocity->GetLargestPossibleRegion().GetSize();
  velocityRegion.SetSize(velocitySize);
  velocity->SetRegions(velocityRegion);
  velocity->FillBuffer(NumericTraits<VectorType>::Zero);

  // Initialize displacement, /phi_{10}
  this->m_OutputTransform->SetVelocityField(velocity);
  this->m_OutputTransform->SetNumberOfIntegrationSteps(m_NumberOfTimeSteps + 2);
  this->m_OutputTransform->SetLowerTimeBound(1.0);
  this->m_OutputTransform->SetUpperTimeBound(0.0);
  this->m_OutputTransform->IntegrateVelocityField();

  // Initialize virtual image using velocity
  typename VirtualImageType::IndexType virtualIndex;
  typename VirtualImageType::SizeType virtualSize;
  typename VirtualImageType::PointType virtualOrigin;
  typename VirtualImageType::SpacingType virtualSpacing;
  typename VirtualImageType::DirectionType virtualDirection;

  for(unsigned int i = 0; i < ImageDimension; i++)
  {
    virtualIndex[i] = velocityIndex[i];
    virtualSize[i] = velocitySize[i];
    virtualOrigin[i] = velocityOrigin[i];
    virtualSpacing[i] = velocitySpacing[i];
    for(unsigned int j = 0; j < ImageDimension; j++)
    {
      virtualDirection(i,j) = velocityDirection(i,j);
    }
  }

  typename VirtualImageType::RegionType virtualRegion(virtualIndex, virtualSize);
  m_VirtualImage->SetRegions(virtualRegion);
  m_VirtualImage->SetOrigin(virtualOrigin);
  m_VirtualImage->SetSpacing(virtualSpacing);
  m_VirtualImage->SetDirection(virtualDirection);

  ContinuousIndex<double, ImageDimension> centerIndex;
  for(unsigned int i = 0; i < ImageDimension; i++)
  {
    centerIndex[i] = (virtualSize[i] - virtualIndex[i] - 1.0) / 2;
  }

  m_VirtualImage->TransformContinuousIndexToPhysicalPoint(centerIndex, m_CenterPoint);

  // Initialize rate, r = 0
  m_Rate->SetRegions(velocityRegion);
  m_Rate->CopyInformation(velocity);
  m_Rate->Allocate();
  m_Rate->FillBuffer(NumericTraits<VirtualPixelType>::Zero);

  // Initialize bias, B = 0
  m_Bias->CopyInformation(m_VirtualImage);
  m_Bias->SetRegions(virtualRegion);
  m_Bias->Allocate();
  m_Bias->FillBuffer(NumericTraits<VirtualPixelType>::Zero);

  // Initialize forward image I(1)
  typedef CastImageFilter<MovingImageType, VirtualImageType> MovingCasterType;
  typename MovingCasterType::Pointer movingCaster = MovingCasterType::New();
  movingCaster->SetInput(this->GetMovingImage());
  movingCaster->Update();
  m_ForwardImage = movingCaster->GetOutput();

  // Initialize forward mask M(1)
  ImageMetricPointer metric = dynamic_cast<ImageMetricType *>(this->m_Metric.GetPointer()); 
  typedef SpatialObjectToImageFilter<MaskType, MaskImageType> MaskToImageType;
  if(metric->GetMovingImageMask())
  {
    typename MaskToImageType::Pointer maskToImage = MaskToImageType::New();
    maskToImage->SetInput(dynamic_cast<const MaskType*>(metric->GetMovingImageMask()));
    maskToImage->SetInsideValue(1);
    maskToImage->SetOutsideValue(0);
    maskToImage->SetSpacing(m_ForwardImage->GetSpacing());
    maskToImage->SetOrigin(m_ForwardImage->GetOrigin());
    maskToImage->SetDirection(m_ForwardImage->GetDirection());
    maskToImage->SetSize(m_ForwardImage->GetLargestPossibleRegion().GetSize());
    maskToImage->Update();

    m_MovingMaskImage = maskToImage->GetOutput(); // M_0
    
    typedef ImageDuplicator<MaskImageType> DuplicatorType;
    typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(m_MovingMaskImage);
    duplicator->Update();
    
    m_ForwardMaskImage = duplicator->GetModifiableOutput(); // M(1) = M_0
  }

  // Initialize fixed mask M_1
  MaskPointer fixedMask;
  if(metric->GetFixedImageMask())
  {
    // Spatial object duplicator not working convert from mask to mask-image and back to mask
    typename MaskToImageType::Pointer maskToImage = MaskToImageType::New();
    maskToImage->SetInput(dynamic_cast<const MaskType*>(metric->GetFixedImageMask()));
    maskToImage->SetInsideValue(1);
    maskToImage->SetOutsideValue(0);
    maskToImage->SetSpacing(fixedImage->GetSpacing());
    maskToImage->SetOrigin(fixedImage->GetOrigin());
    maskToImage->SetDirection(fixedImage->GetDirection());
    maskToImage->SetSize(fixedRegion.GetSize());
    maskToImage->Update();

    fixedMask = MaskType::New();
    fixedMask->SetImage(maskToImage->GetOutput()); // M_1
  }

  // Initialize velocity kernels, K_V, L_V
  InitializeKernels(m_VelocityKernel,m_InverseVelocityKernel,m_RegistrationSmoothness,m_Gamma);

  // Initialize rate kernels, K_R, L_R
  InitializeKernels(m_RateKernel,m_InverseRateKernel,m_BiasSmoothness,m_Gamma);

  // Initialize constants
  m_VoxelVolume = 1;
  for(unsigned int i = 0; i < ImageDimension; i++){ m_VoxelVolume *= virtualSpacing[i]; } // \Delta x
  m_NumberOfTimeSteps = velocity->GetLargestPossibleRegion().GetSize()[ImageDimension]; // J
  m_TimeStep = 1.0/(m_NumberOfTimeSteps - 1); // \Delta t
  m_RecalculateEnergy = true; // v and r have been initialized
  
  typedef CastImageFilter<FixedImageType, VirtualImageType>  FixedCasterType;
  typename FixedCasterType::Pointer fixedCaster = FixedCasterType::New();
  fixedCaster->SetInput(this->GetFixedImage());
  fixedCaster->Update();
  
  m_MinImageEnergy = GetImageEnergy(fixedCaster->GetOutput(), fixedMask);
  m_MaxImageEnergy = GetImageEnergy();
  
  // Disable bias correction if \mu = 0
  if(m_Mu < NumericTraits<double>::epsilon()){ m_UseBias = false; }

  this->InvokeEvent(InitializeEvent());
}

template<typename TFixedImage, typename TMovingImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
CalculateNorm(TimeVaryingImagePointer image)
{
  typedef StatisticsImageFilter<TimeVaryingImageType> CalculatorType;
  typename CalculatorType::Pointer calculator = CalculatorType::New();
  calculator->SetInput(image);
  calculator->Update();
  
  // sumOfSquares = (var(x) + mean(x)^2)*length(x)
  double sumOfSquares = (calculator->GetVariance()+vcl_pow(calculator->GetMean(),2))*image->GetLargestPossibleRegion().GetNumberOfPixels();
  return vcl_sqrt(sumOfSquares*m_VoxelVolume*m_TimeStep);
}

template<typename TFixedImage, typename TMovingImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
CalculateNorm(TimeVaryingFieldPointer field)
{
  typedef VectorMagnitudeImageFilter<TimeVaryingFieldType,TimeVaryingImageType> MagnitudeFilterType;
  typename MagnitudeFilterType::Pointer magnitudeFilter = MagnitudeFilterType::New();
  magnitudeFilter->SetInput(field);
  magnitudeFilter->Update();

  return CalculateNorm(magnitudeFilter->GetOutput());
}


template<typename TFixedImage, typename TMovingImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GetLength()
{
  typename TimeVaryingFieldType::Pointer velocity = this->m_OutputTransform->GetVelocityField();

  typename TimeVaryingImageType::IndexType  index;
  index.Fill(0);

  typename TimeVaryingImageType::SizeType   size = velocity->GetLargestPossibleRegion().GetSize();
  size[ImageDimension] = 1;

  typename TimeVaryingImageType::RegionType region(index,size);

  double length = 0;
  for(unsigned int j = 0; j < m_NumberOfTimeSteps; j++)
  {
    index[ImageDimension] = j;
    region.SetIndex(index);

    typedef ExtractImageFilter<TimeVaryingFieldType, TimeVaryingFieldType> ExtractorType;
    typename ExtractorType::Pointer extractor = ExtractorType::New();
    extractor->SetInput(velocity);                    // v
    extractor->SetExtractionRegion(region);
    extractor->SetDirectionCollapseToIdentity();
    extractor->Update();

    length += CalculateNorm(extractor->GetOutput()); // || v_j || \Delta t
  }

  return length;  // \sum_{j=0}^{J-1} || v_j || \Delta t
}


template<typename TFixedImage, typename TMovingImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GetVelocityEnergy()
{
  return 0.5 * vcl_pow(CalculateNorm(ApplyKernel(m_InverseVelocityKernel,this->m_OutputTransform->GetVelocityField())),2); // 0.5 ||L_V V||^2
}

template<typename TFixedImage, typename TMovingImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GetRateEnergy()
{
  if(m_UseBias)
  {
    return 0.5 * vcl_pow(m_Mu * CalculateNorm(ApplyKernel(m_InverseRateKernel,m_Rate)),2); // 0.5 \mu^2 ||L_R r||^2
  }
  else
  {
    return 0;
  }
}

template<typename TFixedImage, typename TMovingImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GetImageEnergy(VirtualImagePointer movingImage, MaskPointer movingMask)
{    
  typedef CastImageFilter<VirtualImageType, MovingImageType> CasterType;
  typename CasterType::Pointer caster = CasterType::New();
  caster->SetInput(movingImage);                            // I(1)
  caster->Update();

  ImageMetricPointer metric = dynamic_cast<ImageMetricType *>(this->m_Metric.GetPointer()); 
  metric->SetFixedImage(this->GetFixedImage());        // I_1
  metric->SetFixedImageGradientFilter(DefaultFixedImageGradientFilterType::New());
  metric->SetMovingImage(caster->GetOutput());
  metric->SetMovingImageGradientFilter(DefaultMovingImageGradientFilterType::New());
  metric->SetMovingImageMask(movingMask);
  metric->SetVirtualDomainFromImage(m_VirtualImage);
  metric->Initialize();
  
  return 0.5*vcl_pow(m_Sigma,-2) * metric->GetValue() * metric->GetNumberOfValidPoints() * m_VoxelVolume;         // 0.5 \sigma^{-2} ||I(1) - I_1||
}

template<typename TFixedImage, typename TMovingImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GetImageEnergy()
{ 
  MaskPointer forwardMask;
  if(m_ForwardMaskImage)
  {
    forwardMask = MaskType::New();
    forwardMask->SetImage(m_ForwardMaskImage);
  }
  return GetImageEnergy(m_ForwardImage, forwardMask); // I(1)
}

template<typename TFixedImage, typename TMovingImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GetImageEnergyFraction()
{
  double imageEnergyFraction = (GetImageEnergy() - m_MinImageEnergy) / (m_MaxImageEnergy - m_MinImageEnergy);
  if(isnan(imageEnergyFraction)){ return 0; }
  return imageEnergyFraction;
}


template<typename TFixedImage, typename TMovingImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GetEnergy()
{
  if(m_RecalculateEnergy == true)
  {
    m_Energy = GetVelocityEnergy() + GetRateEnergy() + GetImageEnergy(); // E = E_velocity + E_rate + E_image
    m_RecalculateEnergy = false;
  }
  return m_Energy;
}

template<typename TFixedImage, typename TMovingImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
IntegrateRate()
{
  typename TimeVaryingImageType::IndexType  index;
  index.Fill(0);

  typename TimeVaryingImageType::SizeType   size = m_Rate->GetLargestPossibleRegion().GetSize();
  size[ImageDimension] = 0;

  typename TimeVaryingImageType::RegionType region(index,size);

  m_Bias->FillBuffer(NumericTraits<VirtualPixelType>::Zero); // B(0) = 0;

  for(unsigned int j = 1; j < m_NumberOfTimeSteps; j++)
  {
    index[ImageDimension] = j-1;
    region.SetIndex(index);

    typedef ExtractImageFilter<TimeVaryingImageType,VirtualImageType> ExtractorType;
    typename ExtractorType::Pointer extractor = ExtractorType::New();
    extractor->SetInput(m_Rate);                    // r
    extractor->SetExtractionRegion(region);
    extractor->SetDirectionCollapseToIdentity();

    typedef MultiplyImageFilter<VirtualImageType,VirtualImageType> MultiplierType;
    typename MultiplierType::Pointer  multiplier = MultiplierType::New();
    multiplier->SetInput(extractor->GetOutput());   // r(j-1)
    multiplier->SetConstant(m_TimeStep);            // \Delta t

    typedef AddImageFilter<VirtualImageType> AdderType;
    typename AdderType::Pointer adder = AdderType::New();
    adder->SetInput1(multiplier->GetOutput());      // r(j-1) \Delta t
    adder->SetInput2(m_Bias);                       // B(j-1)

    this->m_OutputTransform->SetNumberOfIntegrationSteps(2);
    this->m_OutputTransform->SetLowerTimeBound(j * m_TimeStep);     // t_j
    this->m_OutputTransform->SetUpperTimeBound((j-1) * m_TimeStep); // t_{j-1}
    this->m_OutputTransform->IntegrateVelocityField();

    typedef WrapExtrapolateImageFunction<VirtualImageType, RealType>         ExtrapolatorType;
    typedef ResampleImageFilter<VirtualImageType,VirtualImageType,RealType>  ResamplerType;
    typename ResamplerType::Pointer  resampler = ResamplerType::New();
    resampler->SetInput(adder->GetOutput());                    // r(j-1) \Delta t + B(j-1)
    resampler->SetTransform(this->m_OutputTransform);           // \phi_{j,j-1}
    resampler->UseReferenceImageOn();
    resampler->SetReferenceImage(m_VirtualImage);
    resampler->SetExtrapolator(ExtrapolatorType::New());
    resampler->Update();

    m_Bias = resampler->GetOutput();
  }
}

template<typename TFixedImage, typename TMovingImage>
typename MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::BiasImagePointer
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GetBias()
{
  typedef ResampleImageFilter<VirtualImageType, BiasImageType, RealType>  ResamplerType;
  typename ResamplerType::Pointer resampler = ResamplerType::New();
  resampler->SetInput(m_Bias);   // B(1)
  resampler->UseReferenceImageOn();
  resampler->SetReferenceImage(this->GetFixedImage());
  resampler->Update();

  return resampler->GetOutput();
}

template<typename TFixedImage, typename TMovingImage>
typename MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::FieldPointer
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GetMetricDerivative(FieldPointer field, bool useImageGradients)
{
  FixedImageGradientFilterPointer fixedImageGradientFilter;
  MovingImageGradientFilterPointer movingImageGradientFilter;

  if(useImageGradients)
  {
    fixedImageGradientFilter = DefaultFixedImageGradientFilterType::New(); // \nabla I_1
    movingImageGradientFilter = DefaultMovingImageGradientFilterType::New(); // \nabla I_0
  }
  else
  {
    fixedImageGradientFilter = dynamic_cast<FixedImageGradientFilterType*>(m_FixedImageConstantGradientFilter.GetPointer());    // [1,1,1]
    movingImageGradientFilter = dynamic_cast<MovingImageGradientFilterType*>(m_MovingImageConstantGradientFilter.GetPointer()); // [1,1,1]
  }

  /* Compute metric derivative p(t) \nabla I(t) */
  typedef DisplacementFieldTransform<RealType,ImageDimension> DisplacementFieldTransformType;
  typename DisplacementFieldTransformType::Pointer fieldTransform = DisplacementFieldTransformType::New();
  fieldTransform->SetDisplacementField(field); // \phi_{t1}

  typedef CastImageFilter<VirtualImageType, MovingImageType> CasterType;
  typename CasterType::Pointer caster = CasterType::New();
  caster->SetInput(m_ForwardImage); // I(1)
  caster->Update();

  typename MaskType::Pointer forwardMask;
  if(m_ForwardMaskImage)
  {
    forwardMask = MaskType::New();
    forwardMask->SetImage(m_ForwardMaskImage);
  }

  ImageMetricPointer metric = dynamic_cast<ImageMetricType*>(this->m_Metric.GetPointer()); 
  metric->SetFixedImage(this->GetFixedImage());                    // I_1
  metric->SetFixedTransform(fieldTransform);                       // \phi_{t1}
  metric->SetFixedImageGradientFilter(fixedImageGradientFilter);   // \nabla I_1
  metric->SetMovingImage(caster->GetOutput());                     // I(1)
  metric->SetMovingTransform(fieldTransform);                      // \phi_{t1}
  metric->SetMovingImageGradientFilter(movingImageGradientFilter); // \nabla I_0
  metric->SetMovingImageMask(forwardMask);
  metric->SetVirtualDomainFromImage(m_VirtualImage);
  metric->Initialize(); 

  // Setup metric derivative
  typename MetricDerivativeType::SizeValueType metricDerivativeSize = m_VirtualImage->GetLargestPossibleRegion().GetNumberOfPixels() * ImageDimension;
  MetricDerivativeType metricDerivative(metricDerivativeSize);
  metricDerivative.Fill(NumericTraits<typename MetricDerivativeType::ValueType>::ZeroValue());

  // Get metric derivative
  metric->GetDerivative(metricDerivative); // -dM(I(1) o \phi{t1}, I_1 o \phi{t1})
  VectorType *metricDerivativePointer = reinterpret_cast<VectorType*> (metricDerivative.data_block());

  SizeValueType numberOfPixelsPerTimeStep = m_VirtualImage->GetLargestPossibleRegion().GetNumberOfPixels();

  typedef ImportImageFilter<VectorType, ImageDimension> ImporterType;
  typename ImporterType::Pointer importer = ImporterType::New();
  importer->SetImportPointer(metricDerivativePointer, numberOfPixelsPerTimeStep, false);
  importer->SetRegion(m_VirtualImage->GetLargestPossibleRegion());
  importer->SetOrigin(m_VirtualImage->GetOrigin());
  importer->SetSpacing(m_VirtualImage->GetSpacing());
  importer->SetDirection(m_VirtualImage->GetDirection());
  importer->Update();

  FieldPointer metricDerivativeField = importer->GetOutput();    

  // ITK dense transforms always return identity for jacobian with respect to parameters.  
  // ... so we provide an option to use it here.

  typedef MultiplyImageFilter<FieldType,VirtualImageType>  FieldMultiplierType;

  if(m_UseJacobian)
  {
    typedef DisplacementFieldJacobianDeterminantFilter<FieldType,RealType,VirtualImageType>  JacobianDeterminantFilterType;
    typename JacobianDeterminantFilterType::Pointer jacobianDeterminantFilter = JacobianDeterminantFilterType::New();
    jacobianDeterminantFilter->SetInput(this->m_OutputTransform->GetDisplacementField()); // \phi_{t1}

    typename FieldMultiplierType::Pointer multiplier0 = FieldMultiplierType::New();
    multiplier0->SetInput1(importer->GetOutput());                  // -dM(I(1) o \phi{t1}, I_1 o \phi{t1})
    multiplier0->SetInput2(jacobianDeterminantFilter->GetOutput()); // |D\phi_{t1}|
    multiplier0->Update();

    metricDerivativeField = multiplier0->GetOutput();
  }

  typename FieldMultiplierType::Pointer multiplier1 = FieldMultiplierType::New();
  multiplier1->SetInput(metricDerivativeField);  // -dM(I(1) o \phi{t1}, I_1 o \phi{t1})
  multiplier1->SetConstant(vcl_pow(m_Sigma,-2)); // 0.5 \sigma^{-2}
  multiplier1->Update();
  
  return multiplier1->GetOutput(); // p(t) \nabla I(t) = p(1, \phi{t1})  \nabla I(1, \phi{t1}) = -0.5 \sigma^{-2} -dM(I(1) o \phi{t1}, I_1 o \phi{t1})
}

template<typename TFixedImage, typename TMovingImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
UpdateControls()
{
  typedef JoinSeriesImageFilter<FieldType,TimeVaryingFieldType> FieldJoinerType;
  typename FieldJoinerType::Pointer velocityJoiner = FieldJoinerType::New();

  typedef JoinSeriesImageFilter<VirtualImageType,TimeVaryingImageType> ImageJoinerType;
  typename ImageJoinerType::Pointer rateJoiner = ImageJoinerType::New();

  // For each time step
  for(unsigned int j = 0; j < m_NumberOfTimeSteps; j++)
  {
    double t = j * m_TimeStep;

    //std::cout<<"Before integrate"<<std::endl; /***/
    // Compute reverse mapping, \phi_{t1} by integrating velocity field, v(t).
    if(j == m_NumberOfTimeSteps-1)
    {
      this->m_OutputTransform->GetModifiableDisplacementField()->FillBuffer(NumericTraits<VectorType>::Zero);
    }
    else
    {
      this->m_OutputTransform->SetNumberOfIntegrationSteps((m_NumberOfTimeSteps-1-j) + 2);
      this->m_OutputTransform->SetLowerTimeBound(t);
      this->m_OutputTransform->SetUpperTimeBound(1.0);
      this->m_OutputTransform->IntegrateVelocityField();
    }
    
    //std::cout<<"After integrate"<<std::endl; /***/
    velocityJoiner->PushBackInput(GetMetricDerivative(this->m_OutputTransform->GetDisplacementField(), true)); // p(t) \nabla I(t) =  p(1, \phi{t1})  \nabla I(1, \phi{t1})
    //std::cout<<"After compute derivative"<<std::endl; /***/

    if(m_UseBias)
    {
      typedef VectorIndexSelectionCastImageFilter<FieldType, VirtualImageType> ComponentExtractorType;
      typename ComponentExtractorType::Pointer componentExtractor = ComponentExtractorType::New();
      componentExtractor->SetInput(GetMetricDerivative(this->m_OutputTransform->GetDisplacementField(), false)); // p(t) [1,1,1] = p(t) \nabla I(t) [1,1,1]
      componentExtractor->SetIndex(0);
      componentExtractor->Update();

      rateJoiner->PushBackInput(componentExtractor->GetOutput()); // p(t)
    }

  } // end for j
  velocityJoiner->Update();

  // Compute velocity energy gradient, \nabla_V E = v + K_V [p \nabla I]
  //std::cout<<"Before Apply Kernel"<<std::endl;  /***/
  typedef AddImageFilter<TimeVaryingFieldType> TimeVaryingFieldAdderType;
  typename TimeVaryingFieldAdderType::Pointer adder0 = TimeVaryingFieldAdderType::New();
  adder0->SetInput1(this->m_OutputTransform->GetVelocityField());                 // v
  adder0->SetInput2(ApplyKernel(m_VelocityKernel,velocityJoiner->GetOutput()));   // K_V[p \nabla I]
  adder0->Update();
  TimeVaryingFieldPointer velocityEnergyGradient = adder0->GetOutput();           // \nabla_V E = v + K_V[p \nabla I]
  //std::cout<<"After Apply Kernel"<<std::endl; /***/

  // Compute rate energy gradient \nabla_r E = r - \mu^2 K_R[p]
  typedef MultiplyImageFilter<TimeVaryingImageType,TimeVaryingImageType>  TimeVaryingImageMultiplierType;
  typedef AddImageFilter<TimeVaryingImageType>                            TimeVaryingImageAdderType;
  TimeVaryingImagePointer rateEnergyGradient;
  if(m_UseBias)
  {
    rateJoiner->Update();

    typename TimeVaryingImageMultiplierType::Pointer multiplier1 = TimeVaryingImageMultiplierType::New();
    multiplier1->SetInput(ApplyKernel(m_RateKernel,rateJoiner->GetOutput())); // K_R[p]
    multiplier1->SetConstant(-vcl_pow(m_Mu,-2));     // -\mu^-2

    typename TimeVaryingImageAdderType::Pointer adder1 = TimeVaryingImageAdderType::New();
    adder1->SetInput1(m_Rate);                     // r
    adder1->SetInput2(multiplier1->GetOutput());   // -\mu^2 K_R[p]
    adder1->Update();

    rateEnergyGradient = adder1->GetOutput();      // \nabla_R E = r - \mu^2 K_R[p]
  }

  //std::cout<<"Before GetEnergy"<<std::endl;  /***/
  double                  energyOld = GetEnergy();
  //std::cout<<"After GetEnergy"<<std::endl;  /***/
  TimeVaryingFieldPointer velocityOld = this->m_OutputTransform->GetVelocityField();
  TimeVaryingImagePointer rateOld = m_Rate;

  while(this->GetLearningRate() > m_MinLearningRate && GetImageEnergyFraction() > m_MinImageEnergyFraction)
  {
    // Update velocity, v = v - \epsilon \nabla_V E
    typedef MultiplyImageFilter<TimeVaryingFieldType,TimeVaryingImageType>  TimeVaryingFieldMultiplierType;
    typename TimeVaryingFieldMultiplierType::Pointer multiplier2 = TimeVaryingFieldMultiplierType::New();
    multiplier2->SetInput(velocityEnergyGradient);                   // \nabla_V E
    multiplier2->SetConstant(-this->GetLearningRate());              // -\epsilon

    typename TimeVaryingFieldAdderType::Pointer adder2 = TimeVaryingFieldAdderType::New();
    adder2->SetInput1(this->m_OutputTransform->GetVelocityField());   // v
    adder2->SetInput2(multiplier2->GetOutput());                      // -\epsilon \nabla_V E
    adder2->Update();

    this->m_OutputTransform->SetVelocityField(adder2->GetOutput());  // v = v - \epsilon \nabla_V E

    // Compute forward mapping \phi{10} by integrating velocity field v(t)
    this->m_OutputTransform->SetNumberOfIntegrationSteps((m_NumberOfTimeSteps -1) + 2);
    this->m_OutputTransform->SetLowerTimeBound(1.0);
    this->m_OutputTransform->SetUpperTimeBound(0.0);
    this->m_OutputTransform->IntegrateVelocityField();

    typedef DisplacementFieldTransform<RealType,ImageDimension> DisplacementFieldTransformType;
    typename DisplacementFieldTransformType::Pointer transform = DisplacementFieldTransformType::New();
    transform->SetDisplacementField(this->m_OutputTransform->GetDisplacementField()); // \phi_{t1}
   
    // Compute forward image I(1) = I_0 o \phi_{10} + B(1)
    typedef WrapExtrapolateImageFunction<MovingImageType, RealType>         ExtrapolatorType;
    typedef ResampleImageFilter<MovingImageType,VirtualImageType,RealType>  MovingResamplerType;
    typename MovingResamplerType::Pointer resampler = MovingResamplerType::New();
    resampler->SetInput(this->GetMovingImage());   // I_0
    resampler->SetTransform(transform);            // \phi_{t0}
    resampler->UseReferenceImageOn();
    resampler->SetReferenceImage(this->GetFixedImage());
    resampler->SetExtrapolator(ExtrapolatorType::New());
    resampler->Update();

    m_ForwardImage = resampler->GetOutput();       // I_0 o \phi_{10}
    
    // Compute forward mask M(1) = M_0 o \phi{1_0} 
    if(m_ForwardMaskImage)
    {
      typedef NearestNeighborInterpolateImageFunction<MaskImageType, RealType>      MaskInterpolatorType;
      typename MaskInterpolatorType::Pointer maskInterpolator = MaskInterpolatorType::New();

      typedef WrapExtrapolateImageFunction<MaskImageType, RealType>         MaskExtrapolatorType;
      typename MaskExtrapolatorType::Pointer maskExtrapolator = MaskExtrapolatorType::New();
      maskExtrapolator->SetInterpolator(maskInterpolator);

      typedef ResampleImageFilter<MaskImageType,MaskImageType,RealType>  MaskResamplerType;
      typename MaskResamplerType::Pointer maskResampler = MaskResamplerType::New();
      maskResampler->SetInput(m_MovingMaskImage);  // M_0
      maskResampler->SetTransform(transform);      // \phi_{10}
      maskResampler->UseReferenceImageOn();
      maskResampler->SetReferenceImage(this->GetFixedImage());
      maskResampler->SetInterpolator(maskInterpolator);
      maskResampler->SetExtrapolator(maskExtrapolator);
      maskResampler->Update();

      m_ForwardMaskImage = maskResampler->GetOutput(); // M_0 o \phi_{10} 
    }
    
    if(m_UseBias)
    {
      // Update rate, r = r - \epsilon \nabla_R E
      typename TimeVaryingImageMultiplierType::Pointer multiplier3 = TimeVaryingImageMultiplierType::New();
      multiplier3->SetInput(rateEnergyGradient);            // \nabla_R E
      multiplier3->SetConstant(-this->GetLearningRate());   // -\epsilon

      typename TimeVaryingImageAdderType::Pointer adder3 = TimeVaryingImageAdderType::New();
      adder3->SetInput1(m_Rate);                    // r
      adder3->SetInput2(multiplier3->GetOutput());  // -\epsilon \nabla_R E
      adder3->Update();

      m_Rate = adder3->GetOutput(); // r = r - \epsilon \nabla_R E  */
      IntegrateRate();

      typedef AddImageFilter<VirtualImageType>   AdderType;
      typename AdderType::Pointer biasAdder = AdderType::New();
      biasAdder->SetInput1(m_ForwardImage);    // I_0 o \phi_{10}
      biasAdder->SetInput2(GetBias());         // B(1)
      biasAdder->Update();
    
      m_ForwardImage = biasAdder->GetOutput(); // I_0 o \phi_{10} + B(1)
    }

    m_RecalculateEnergy = true;
   
    typename VirtualImageType::IndexType centerIndex;
    m_ForwardImage->TransformPhysicalPointToIndex(transform->TransformPoint(m_CenterPoint), centerIndex);
    bool centerIsInside = m_ForwardImage->GetLargestPossibleRegion().IsInside(centerIndex);
    if(!centerIsInside || GetEnergy() > energyOld)  // If energy increased or transformed center point of reference image is outside input image domain
    {
      // ...restore the controls to their previous values and decrease learning rate
      this->SetLearningRate(0.5*this->GetLearningRate());
      this->m_OutputTransform->SetVelocityField(velocityOld);
      m_Rate = rateOld;
      m_RecalculateEnergy = true;
    }
    else // If energy decreased...
    {
      // ...slightly increase learning rate
      const double learningAcceleration = 1.1; // 1.03
      this->SetLearningRate(learningAcceleration*this->GetLearningRate());
      return;
    }

  }
  
  m_IsConverged = true;
}

template<typename TFixedImage, typename TMovingImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
StartOptimization()
{
  this->InvokeEvent(StartEvent());
  for(this->m_CurrentIteration = 0; this->m_CurrentIteration < m_NumberOfIterations; this->m_CurrentIteration++)
  {
    UpdateControls();
    if(this->m_IsConverged){ break; }
    this->InvokeEvent(IterationEvent());
  }
}

template<typename TFixedImage, typename TMovingImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GenerateData()
{
  this->m_OutputTransform->UseInverseOff();
  Initialize();
  StartOptimization();
  this->m_OutputTransform->UseInverseOn();

  // Integrate rate to get final bias, B(1)  
  if(m_UseBias) { IntegrateRate(); }

  // Integrate velocity to get final displacement, \phi_10
  this->m_OutputTransform->SetNumberOfIntegrationSteps(m_NumberOfTimeSteps + 2);
  this->m_OutputTransform->SetLowerTimeBound(1.0);
  this->m_OutputTransform->SetUpperTimeBound(0.0);
  this->m_OutputTransform->IntegrateVelocityField();
  this->GetTransformOutput()->Set(this->m_OutputTransform);

  this->InvokeEvent(EndEvent());
}

template<typename TFixedImage, typename TMovingImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
PrintSelf(std::ostream& os, Indent indent ) const
{
  ProcessObject::PrintSelf(os, indent);
  os<<indent<<"Velocity Smoothness: " <<m_RegistrationSmoothness<<std::endl;
  os<<indent<<"Bias Smoothness: "<<m_BiasSmoothness<<std::endl;
}


} // End namespace itk


#endif
