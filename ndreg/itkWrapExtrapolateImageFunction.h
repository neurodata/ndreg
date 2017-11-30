#ifndef itkWrapExtrapolateImageFunction_h
#define itkWrapExtrapolateImageFunction_h

#include "itkExtrapolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
namespace itk
{
/** \class WrapExtrapolateImageFunction
 * \brief Wrap extrapolation of a scalar image.
 *
 * WrapExtrapolateImageFunction wraps specified point, continuous index or index to obtain
 * the intensity of pixel within the image buffer.
 *
 * This class is templated
 * over the input image type and the coordinate representation type
 * (e.g. float or double).
 *
 * \ingroup ImageFunctions
 * \ingroup ITKImageFunction
 */
template< typename TInputImage, typename TCoordRep = float >
class WrapExtrapolateImageFunction:
  public ExtrapolateImageFunction< TInputImage, TCoordRep >
{
public:
  /** Standard class typedefs. */
  typedef WrapExtrapolateImageFunction            Self;
  typedef ExtrapolateImageFunction< TInputImage, TCoordRep > Superclass;
  typedef SmartPointer< Self >                               Pointer;
  typedef SmartPointer< const Self >                         ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(WrapExtrapolateImageFunction,
               InterpolateImageFunction);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** OutputType typedef support. */
  typedef typename Superclass::OutputType OutputType;

  /** InputImageType typedef support. */
  typedef typename Superclass::InputImageType InputImageType;

  /** Dimension underlying input image. */
  itkStaticConstMacro(ImageDimension, unsigned int, Superclass::ImageDimension);

  /** Index typedef support. */
  typedef typename Superclass::IndexType      IndexType;
  typedef typename IndexType::IndexValueType  IndexValueType;

  /** ContinuousIndex typedef support. */
  typedef typename Superclass::ContinuousIndexType ContinuousIndexType;

  /** Interpolator typedef support. */
  typedef InterpolateImageFunction<InputImageType, TCoordRep> InterpolatorType;
  typedef typename InterpolatorType::Pointer InterpolatorPointerType;

  typedef LinearInterpolateImageFunction<InputImageType, TCoordRep> LinearInterpolatorType;
  typedef typename LinearInterpolatorType::Pointer LinearInterpolatorPointerType;

  itkGetModifiableObjectMacro(Interpolator, InterpolatorType);

  /** Evaluate the function at a ContinuousIndex position
   *
   * Returns the extrapolated image intensity at a
   * specified position
   */
  virtual OutputType EvaluateAtContinuousIndex(
    const ContinuousIndexType & index) const ITK_OVERRIDE
  {
    
    ContinuousIndexType nindex;

    for ( unsigned int j = 0; j < ImageDimension; j++ )
    {
      nindex[j] = index[j];
      typename ContinuousIndexType::ValueType size = this->GetEndContinuousIndex()[j] - this->GetStartContinuousIndex()[j];

      while(nindex[j] > this->GetEndIndex()[j])
      {
        nindex[j] -= size;
      }
      while(nindex[j] < this->GetStartIndex()[j])
      {
        nindex[j] += size;
      }
    }

    
    return static_cast< OutputType >( m_Interpolator->EvaluateAtContinuousIndex(nindex) );
  }


  void SetInputImage(const InputImageType*ptr)
  {
    Superclass::SetInputImage(ptr);
    m_Interpolator->SetInputImage(this->GetInputImage());
  }

  void SetInterpolator(InterpolatorType* ptr)
  {
    m_Interpolator = dynamic_cast<InterpolatorType*>(ptr);
    if(ptr != ITK_NULLPTR)
    {
      m_Interpolator->SetInputImage(this->GetInputImage());
    }
  }

  /** Evaluate the function at a ContinuousIndex position
   *
   * Returns the extrapolated image intensity at a
   * specified position.
   */
  virtual OutputType EvaluateAtIndex(
    const IndexType & index) const ITK_OVERRIDE
  {
    IndexType nindex;

    for ( unsigned int j = 0; j < ImageDimension; j++ )
    {
      nindex[j] = index[j];

      typename IndexType::IndexValueType size = this->GetEndIndex()[j] - this->GetStartIndex()[j] + 1;
      
      while(nindex[j] > this->GetEndIndex()[j])
      {
        nindex[j] -= size;
      }
      while(nindex[j] < this->GetStartIndex()[j])
      {
        nindex[j] += size;
      }
    }
    return static_cast< OutputType >( this->GetInputImage()->GetPixel(nindex) );
  }

protected:
  WrapExtrapolateImageFunction()
  { m_Interpolator = dynamic_cast<InterpolatorType*>(LinearInterpolatorType::New().GetPointer()); }
  ~WrapExtrapolateImageFunction(){}
  void PrintSelf(std::ostream & os, Indent indent) const ITK_OVERRIDE
  {
    Superclass::PrintSelf(os, indent);
    os << indent << "Interpolator: " << this->m_Interpolator << std::endl;
  }

private: 
  WrapExtrapolateImageFunction(const Self &) ITK_DELETE_FUNCTION;
  void operator=(const Self &) ITK_DELETE_FUNCTION;
  
  InterpolatorPointerType m_Interpolator;
};
} // end namespace itk

#endif
