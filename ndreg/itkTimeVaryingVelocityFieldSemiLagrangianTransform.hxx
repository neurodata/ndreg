#ifndef itkTimeVaryingVelocityFieldSemiLagrangianTransform_hxx
#define itkTimeVaryingVelocityFieldSemiLagrangianTransform_hxx

#include "itkTimeVaryingVelocityFieldSemiLagrangianTransform.h"
#include "itkTimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter.h"

namespace itk
{


template<typename TParametersValueType, unsigned int NDimensions>
TimeVaryingVelocityFieldSemiLagrangianTransform<TParametersValueType, NDimensions>
::TimeVaryingVelocityFieldSemiLagrangianTransform()
{
  m_UseInverse = true;
}


template<typename TParametersValueType, unsigned int NDimensions>
void
TimeVaryingVelocityFieldSemiLagrangianTransform<TParametersValueType, NDimensions>
::IntegrateVelocityField()
{
  if( this->GetVelocityField() )
  {
    typedef TimeVaryingVelocityFieldSemiLagrangianIntegrationImageFilter
      <VelocityFieldType, DisplacementFieldType> IntegratorType;

    typename IntegratorType::Pointer integrator = IntegratorType::New();
    integrator->SetInput( this->GetVelocityField() );
    integrator->SetLowerTimeBound( this->GetLowerTimeBound() );
    integrator->SetUpperTimeBound( this->GetUpperTimeBound() );

    if( this->GetVelocityFieldInterpolator() )
      {
      integrator->SetVelocityFieldInterpolator( this->GetModifiableVelocityFieldInterpolator() );
      }

    integrator->SetNumberOfIntegrationSteps( this->GetNumberOfIntegrationSteps() );
    integrator->Update();

    typename DisplacementFieldType::Pointer displacementField = integrator->GetOutput();
    displacementField->DisconnectPipeline();

    this->SetDisplacementField( displacementField );
    this->GetModifiableInterpolator()->SetInputImage( displacementField );

    if(m_UseInverse)
    {
      typename IntegratorType::Pointer inverseIntegrator = IntegratorType::New();
      inverseIntegrator->SetInput( this->GetVelocityField() );
      inverseIntegrator->SetLowerTimeBound( this->GetUpperTimeBound() );
      inverseIntegrator->SetUpperTimeBound( this->GetLowerTimeBound() );
      if( !this->GetVelocityFieldInterpolator() )
      {
        inverseIntegrator->SetVelocityFieldInterpolator( this->GetModifiableVelocityFieldInterpolator() );
      }

      inverseIntegrator->SetNumberOfIntegrationSteps( this->GetNumberOfIntegrationSteps() );
      inverseIntegrator->Update();

      typename DisplacementFieldType::Pointer inverseDisplacementField = inverseIntegrator->GetOutput();
      inverseDisplacementField->DisconnectPipeline();

      this->SetInverseDisplacementField( inverseDisplacementField );
    }

  }
  else
  {
    itkExceptionMacro( "The velocity field does not exist." );
  }
}

} // namespace itk

#endif
