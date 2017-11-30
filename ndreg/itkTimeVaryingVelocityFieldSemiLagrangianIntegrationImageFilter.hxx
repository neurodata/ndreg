#ifndef itkTimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter_hxx
#define itkTimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter_hxx

#include "itkTimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter.h"

#include "itkImageRegionIteratorWithIndex.h"
#include "itkVectorLinearInterpolateImageFunction.h"

namespace itk
{

/*
 * TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter class definitions
 */
template<typename TTimeVaryingVelocityField, typename TDisplacementField>
TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter
  <TTimeVaryingVelocityField, TDisplacementField>
::TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter()
{
  this->m_LowerTimeBound =  0.0,
  this->m_UpperTimeBound = 1.0,
  this->m_NumberOfIntegrationSteps = 100;
  this->m_NumberOfIterations = 3;
  this->m_NumberOfTimePoints = 0;
  this->SetNumberOfRequiredInputs( 1 );

  if( InputImageDimension - 1 != OutputImageDimension )
    {
    itkExceptionMacro( "The time-varying velocity field (input) should have "
      << "dimensionality of 1 greater than the deformation field (output). " );
    }

  typedef VectorLinearInterpolateImageFunction<TimeVaryingVelocityFieldType, ScalarType> DefaultVelocityFieldInterpolatorType;
  this->SetVelocityFieldInterpolator(DefaultVelocityFieldInterpolatorType::New());

  typedef WrapExtrapolateImageFunction<TimeVaryingVelocityFieldType, ScalarType> DefaultVelocityFieldExtrapolatorType;
  this->SetVelocityFieldExtrapolator(DefaultVelocityFieldExtrapolatorType::New());

  typedef VectorLinearInterpolateImageFunction<DisplacementFieldType, ScalarType> DefaultDisplacementFieldInterpolatorType;
  this->SetDisplacementFieldInterpolator(DefaultDisplacementFieldInterpolatorType::New());

  typedef WrapExtrapolateImageFunction<DisplacementFieldType, ScalarType> DefaultDisplacementFieldExtrapolatorType;
  this->SetDisplacementFieldExtrapolator(DefaultDisplacementFieldExtrapolatorType::New());
}

template<typename TTimeVaryingVelocityField, typename TDisplacementField>
TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter
  <TTimeVaryingVelocityField, TDisplacementField>
::~TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter()
{
}

template<typename TTimeVaryingVelocityField, typename TDisplacementField>
void
TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter
  <TTimeVaryingVelocityField, TDisplacementField>
::BeforeThreadedGenerateData()
{
  Superclass::BeforeThreadedGenerateData();
  this->m_VelocityFieldExtrapolator->SetInputImage( this->GetInput() );

  if( !this->m_InitialDiffeomorphism.IsNull() )
  {
    this->m_DisplacementFieldExtrapolator->SetInputImage( this->m_InitialDiffeomorphism );
  }

  // Find origin of time dimension
  const TimeVaryingVelocityFieldType* inputField = this->GetInput();
  typename TimeVaryingVelocityFieldType::PointType spaceTimeOrigin = inputField->GetOrigin();
  m_TimeOrigin = spaceTimeOrigin[InputImageDimension-1];

  // Find end of time dimension
  typedef typename TimeVaryingVelocityFieldType::RegionType  RegionType;
  RegionType region = inputField->GetLargestPossibleRegion();
  typename RegionType::IndexType lastIndex = region.GetIndex();
  typename RegionType::SizeType size = region.GetSize();
  for( unsigned k = 0; k < InputImageDimension; k++ ){ lastIndex[k] += ( size[k] - 1 );  }

  typename TimeVaryingVelocityFieldType::PointType spaceTimeEnd;
  inputField->TransformIndexToPhysicalPoint( lastIndex, spaceTimeEnd );
  const RealType timeEnd = spaceTimeEnd[InputImageDimension-1];

  // Find span of time dimension
  m_TimeSpan = timeEnd - m_TimeOrigin;

  // Calculate the delta time used for integration
  m_DeltaTime = (this->m_UpperTimeBound - this->m_LowerTimeBound ) / static_cast<RealType>(this->m_NumberOfIntegrationSteps);
}

template<typename TTimeVaryingVelocityField, typename TDisplacementField>
void
TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter
<TTimeVaryingVelocityField, TDisplacementField>
::ThreadedGenerateData( const OutputRegionType &region, ThreadIdType itkNotUsed( threadId ) )
{
  if( Math::ExactlyEquals( this->m_LowerTimeBound, this->m_UpperTimeBound ) )
  {
    return;
  }

  if( this->m_NumberOfIntegrationSteps == 0 )
  {
    return;
  }

  const TimeVaryingVelocityFieldType * inputField = this->GetInput();

  typename DisplacementFieldType::Pointer outputField = this->GetOutput();

  ImageRegionIteratorWithIndex<DisplacementFieldType> It( outputField, region );

  for( It.GoToBegin(); !It.IsAtEnd(); ++It )
  {
    PointType point;
    outputField->TransformIndexToPhysicalPoint( It.GetIndex(), point );
    VectorType displacement = this->IntegrateVelocityAtPoint( point, inputField );
    It.Set( displacement );
  }

}


template<typename TTimeVaryingVelocityField, typename TDisplacementField>
typename TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter
  <TTimeVaryingVelocityField, TDisplacementField>::VectorType
TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter
  <TTimeVaryingVelocityField, TDisplacementField>
::IntegrateVelocityAtPoint( const PointType & initialSpatialPoint,
                            const TimeVaryingVelocityFieldType *inputField )
{
  // Set initial position
  PointType currentSpatialPoint = initialSpatialPoint;
  if( !this->m_InitialDiffeomorphism.IsNull() )
  {
    if( this->GetDisplacementFieldInterpolator()->IsInsideBuffer( currentSpatialPoint ) )
    { currentSpatialPoint += this->GetDisplacementFieldInterpolator()->Evaluate( currentSpatialPoint ); }
    else
    { currentSpatialPoint += this->GetDisplacementFieldExtrapolator()->Evaluate( currentSpatialPoint ); }
  }

  // Set initial time
  RealType timePoint = this->m_LowerTimeBound + 0.5*m_DeltaTime;

  // Advect point
  for(unsigned int j = 0; j < this->m_NumberOfIntegrationSteps; j++, timePoint+=m_DeltaTime)
  {
    VectorType displacement; displacement.Fill(0);
    PointType  spatialPoint = currentSpatialPoint;
    for(unsigned int i = 0; i < this->m_NumberOfIterations; i++)
    {
      displacement = displacement * 0.5; // Don't step too far!
      spatialPoint = currentSpatialPoint + displacement;

      typename TimeVaryingVelocityFieldType::PointType spaceTimePoint;
      for(unsigned int k = 0; k < OutputImageDimension; k++){ spaceTimePoint[k] = spatialPoint[k]; }
      spaceTimePoint[OutputImageDimension] = m_TimeSpan*timePoint + m_TimeOrigin;

      VectorType velocity;
      if(this->GetVelocityFieldInterpolator()->IsInsideBuffer(spaceTimePoint))
      { velocity = this->GetVelocityFieldInterpolator()->Evaluate(spaceTimePoint); }
      else
      { velocity = this->GetVelocityFieldExtrapolator()->Evaluate(spaceTimePoint); }

      displacement = velocity*m_DeltaTime;
    }
    currentSpatialPoint += displacement;
  }

  return currentSpatialPoint.GetVectorFromOrigin() - initialSpatialPoint.GetVectorFromOrigin();
}

template<typename TTimeVaryingVelocityField, typename TDisplacementField>
void
TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter<TTimeVaryingVelocityField, TDisplacementField>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "VelocityFieldExtrapolator: " << this->m_VelocityFieldExtrapolator << std::endl;
  os << indent << "DisplacementFieldExtrapolator: " << this->m_DisplacementFieldExtrapolator << std::endl;
}

}  //end namespace itk

#endif
