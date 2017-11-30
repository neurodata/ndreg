#ifndef itkTimeVaryingVelocityFieldSemiLagrangianTransform_h
#define itkTimeVaryingVelocityFieldSemiLagrangianTransform_h

#include "itkTimeVaryingVelocityFieldTransform.h"

namespace itk
{

/** \class TimeVaryingVelocityFieldSemiLagrangianTransform
 * \brief Transform objects based on integration of a time-varying velocity
 * field using Semi-Lagrangian advection.
 *
 *
 * \ingroup Transforms
 * \ingroup ITKDisplacementField
 */
template<typename TParametersValueType, unsigned int NDimensions>
class TimeVaryingVelocityFieldSemiLagrangianTransform :
  public TimeVaryingVelocityFieldTransform<TParametersValueType, NDimensions>
{
public:
  /** Standard class typedefs. */
  typedef TimeVaryingVelocityFieldSemiLagrangianTransform                         Self;
  typedef TimeVaryingVelocityFieldTransform<TParametersValueType, NDimensions> Superclass;
  typedef SmartPointer<Self>                                        Pointer;
  typedef SmartPointer<const Self>                                  ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( TimeVaryingVelocityFieldSemiLagrangianTransform, TimeVaryingVelocityFieldTransform );

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro( Self );

  /** InverseTransform type. */
  typedef typename Superclass::InverseTransformBasePointer InverseTransformBasePointer;

  /** Interpolator types.*/
  typedef typename Superclass::InterpolatorType                     InterpolatorType;

  /** Field types. */
  typedef typename Superclass::DisplacementFieldType                DisplacementFieldType;
  typedef typename Superclass::VelocityFieldType                    VelocityFieldType;

  typedef typename Superclass::TimeVaryingVelocityFieldType          TimeVaryingVelocityFieldType;
  typedef typename Superclass::TimeVaryingVelocityFieldPointer      TimeVaryingVelocityFieldPointer;

  /** Scalar type. */
  typedef typename Superclass::ScalarType              ScalarType;

  /** Type of the input parameters. */
  typedef typename Superclass::ParametersType           ParametersType;
  typedef typename Superclass::ParametersValueType      ParametersValueType;
  typedef typename Superclass::FixedParametersType      FixedParametersType;
  typedef typename Superclass::FixedParametersValueType FixedParametersValueType;
  typedef typename Superclass::NumberOfParametersType   NumberOfParametersType;

  /** Derivative type */
  typedef typename Superclass::DerivativeType          DerivativeType;
  typedef typename Superclass::TransformPointer        TransformPointer;

  /** Trigger the computation of the displacement field by integrating
   * the time-varying velocity field. */
  virtual void IntegrateVelocityField() ITK_OVERRIDE;
  itkBooleanMacro(UseInverse);
  itkSetMacro(UseInverse, bool);
  itkGetConstMacro(UseInverse, bool);

protected:
  TimeVaryingVelocityFieldSemiLagrangianTransform();
  virtual ~TimeVaryingVelocityFieldSemiLagrangianTransform(){};

private:
  TimeVaryingVelocityFieldSemiLagrangianTransform( const Self& ) ITK_DELETE_FUNCTION;
  void operator=( const Self& ) ITK_DELETE_FUNCTION;
  bool m_UseInverse;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkTimeVaryingVelocityFieldSemiLagrangianTransform.hxx"
#endif

#endif // itkTimeVaryingVelocityFieldSemiLagrangianTransform_h
