#ifndef itkTimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter_h
#define itkTimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter_h

#include "itkExtrapolateImageFunction.h"
#include "itkWrapExtrapolateImageFunction.h"
#include "itkTimeVaryingVelocityFieldIntegrationImageFilter.h"

namespace itk
{
/**
 * \class TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter
 * \brief Integrate a time-varying velocity field using a Semi-Lagrangian scheme.
 *
 *
 * \warning The output deformation field needs to have dimensionality of 1
 * less than the input time-varying velocity field.
 *
 * \ingroup ITKDisplacementField
 */
template<typename TTimeVaryingVelocityField, typename TDisplacementField =
 Image<typename TTimeVaryingVelocityField::PixelType,
 TTimeVaryingVelocityField::ImageDimension - 1> >
class TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter :
  public TimeVaryingVelocityFieldIntegrationImageFilter<TTimeVaryingVelocityField, TDisplacementField>
{
public:
  typedef TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter  Self;
  typedef TimeVaryingVelocityFieldIntegrationImageFilter
    <TTimeVaryingVelocityField, TDisplacementField>       Superclass;
  typedef SmartPointer<Self>                              Pointer;
  typedef SmartPointer<const Self>                        ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information ( and related methods ) */
  itkTypeMacro( TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter, TimeVaryingVelocityFieldIntegrationImageFilter );

  /**
   * Dimensionality of input data is assumed to be one more than the output
   * data the same. */


  itkStaticConstMacro( InputImageDimension, unsigned int,
    TTimeVaryingVelocityField::ImageDimension );

  itkStaticConstMacro( OutputImageDimension, unsigned int,
    TDisplacementField::ImageDimension );

  typedef TTimeVaryingVelocityField                   TimeVaryingVelocityFieldType;
  typedef TDisplacementField                          DisplacementFieldType;
  typedef typename Superclass::VectorType             VectorType;
  typedef typename Superclass::RealType               RealType;
  typedef typename Superclass::ScalarType             ScalarType;
  typedef typename Superclass::PointType              PointType;
  typedef typename DisplacementFieldType::RegionType  OutputRegionType;

  typedef typename Superclass::VelocityFieldInterpolatorPointer VelocityFieldInterpolatorPointer;

  typedef ExtrapolateImageFunction<TimeVaryingVelocityFieldType, ScalarType> VelocityFieldExtrapolatorType;
  typedef typename VelocityFieldExtrapolatorType::Pointer VelocityFieldExtrapolatorPointer;

  typedef ExtrapolateImageFunction<DisplacementFieldType, ScalarType> DisplacementFieldExtrapolatorType;
  typedef typename DisplacementFieldExtrapolatorType::Pointer DisplacementFieldExtrapolatorPointer;

  /**
   * Get/Set the time-varying velocity field extrapolator.  Default = linear. 
   */
  itkSetObjectMacro( VelocityFieldExtrapolator, VelocityFieldExtrapolatorType );
  itkGetModifiableObjectMacro(VelocityFieldExtrapolator, VelocityFieldExtrapolatorType );

  /**
   * Get/Set the deformation field extrapolator for the initial diffeomorphism
   * (if set).
   */
  itkSetObjectMacro( DisplacementFieldExtrapolator, DisplacementFieldExtrapolatorType );
  itkGetModifiableObjectMacro(DisplacementFieldExtrapolator, DisplacementFieldExtrapolatorType );

  /**
   * Get the number of iterations used per integration steps.
   */
  itkGetConstMacro( NumberOfIterations, unsigned int );


protected:
  TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter();
  ~TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter();

  void PrintSelf( std::ostream & os, Indent indent ) const ITK_OVERRIDE;
  virtual void BeforeThreadedGenerateData() ITK_OVERRIDE;
  virtual void ThreadedGenerateData( const OutputRegionType &, ThreadIdType ) ITK_OVERRIDE;
  VectorType IntegrateVelocityAtPoint( const PointType &initialSpatialPoint, const TimeVaryingVelocityFieldType * inputField );

  DisplacementFieldExtrapolatorPointer      m_DisplacementFieldExtrapolator;
 
private:
  TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter( const Self & ) ITK_DELETE_FUNCTION;
  void operator=( const Self & ) ITK_DELETE_FUNCTION;

  VelocityFieldInterpolatorPointer          m_VelocityFieldInterpolator;
  VelocityFieldExtrapolatorPointer          m_VelocityFieldExtrapolator;
  unsigned int                              m_NumberOfIterations;
  RealType                                  m_DeltaTime;
  RealType                                  m_TimeSpan;
  RealType                                  m_TimeOrigin;
};
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkTimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter.hxx"
#endif

#endif
