#include <iomanip>   // setprecision()
#include <algorithm> // min(), max()
#include "itkImageMaskSpatialObject.h"
#include "itkNumericTraits.h"
#include "itkCommand.h"
#include "itkTimeProbe.h"
#include "itkImage.h"
#include "itkCommandLineArgumentParser.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkClampImageFilter.h"
#include "itkBSplineKernelFunction.h"
#include "itkGridImageSource.h"
#include "itkWrapExtrapolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkDisplacementFieldTransform.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkCheckerBoardImageFilter.h"
#include "itkMetamorphosisImageRegistrationMethodv4.h"
#include "itkWrapExtrapolateImageFunction.h"
#include "itkImageToImageMetricv4.h"
using namespace std;

typedef itk::CommandLineArgumentParser  ParserType;
template<typename TMetamorphosis> class MetamorphosisObserver;
template<typename TImage> int Metamorphosis(typename TImage::Pointer fixedImage, typename ParserType::Pointer parser);

int main(int argc, char* argv[])
{
  /** Check command line arguments */
  ParserType::Pointer parser = ParserType::New();
  parser->SetCommandLineArguments(argc,argv);

  if( !(parser->ArgumentExists("--in") && parser->ArgumentExists("--ref") && parser->ArgumentExists("--out")) )
  {
    cerr<<"Usage:"<<endl;
    cerr<<"\t"<<argv[0]<<" --in InputPath1 ... InputPathN --ref ReferencePath1 ... ReferencePathN --out OutputPath"<<endl;
    cerr<<"\t\t[ --inmask inMaskPath"<<endl;
    cerr<<"\t\t  --refmask refMaskPath"<<endl;
    cerr<<"\t\t  --outmask outMaskPath"<<endl;
    cerr<<"\t\t  --field OutputDisplacementFieldPath"<<endl;
    cerr<<"\t\t  --invfield OutputInverseDisplacementFieldPath"<<endl;
    cerr<<"\t\t  --bias OutputBiasPath"<<endl;
    cerr<<"\t\t  --grid OutputGridPath"<<endl;
    cerr<<"\t\t  --gridstep GridStep"<<endl;
    cerr<<"\t\t  --checker OutputCheckerBoardPath"<<endl;
    cerr<<"\t\t  --scale Scale"<<endl;
    cerr<<"\t\t  --alpha RegistrationSmoothness"<<endl;
    cerr<<"\t\t  --beta BiasSmoothness"<<endl;
    cerr<<"\t\t  --sigma Sigma"<<endl;
    cerr<<"\t\t  --mu Mu"<<endl;
    cerr<<"\t\t\t 0 = Disable bias correction"<<endl;
    cerr<<"\t\t  --epsilon LearningRate"<<endl;
    cerr<<"\t\t  --epsilonmin MinLearningRate"<<endl;
    cerr<<"\t\t  --fraction MinimumInitialEnergyFraction"<<endl;
    cerr<<"\t\t  --steps NumberOfTimesteps"<<endl;
    cerr<<"\t\t  --iterations MaximumNumberOfIterations"<<endl;
    cerr<<"\t\t  --cost CostFunction"<<endl;
    cerr<<"\t\t\t 0 = Mean Square Error"<<endl;
    cerr<<"\t\t\t 1 = Mattes Mutual Information"<<endl;
    cerr<<"\t\t  --bins Number of bins used with MI"<<endl;
    cerr<<"\t\t  --verbose ]"<<endl;
    return EXIT_FAILURE;
  }

  /** Read reference image as 3D volume */
  const unsigned int numberOfDimensions = 3;
  typedef float      PixelType;
  typedef itk::Image<PixelType,2>                    Image2DType;
  typedef itk::Image<PixelType,3>                    Image3DType;

  //string referencePath;
  //parser->GetCommandLineArgument("--ref",referencePath);
  std::vector<std::string> referencePaths;
  parser->GetCommandLineArgument("--ref",referencePaths);
  
  //initialize vector of image pointers
  //std::vector<Image3DType::Pointer> referenceImages;
  
  typedef itk::ImageFileReader<Image3DType> ReaderType;
  ReaderType::Pointer referenceReader = ReaderType::New();
  Image3DType::Pointer firstReferenceImage;
  Image3DType::DirectionType direction; direction.SetIdentity();
  //
  
  //for loop to read each image
  for(std::vector<std::string>::const_iterator i = referencePaths.begin(); i != referencePaths.end(); ++i)
  {
      //std::cout<<*i<<std::endl;
      referenceReader->SetFileName(*i);
      try
      {
          referenceReader->Update();
      }
      catch(itk::ExceptionObject& exceptionObject)
      {
          cerr<<"Error: Could not read reference image: "<<*i<<endl;
          cerr<<exceptionObject<<endl;
          return EXIT_FAILURE;
      }
      if(i==referencePaths.begin())
      {
         //std::cout<<"first iteration"<<std::endl;
          firstReferenceImage = referenceReader->GetOutput();
          firstReferenceImage->SetDirection(direction);
          break;
      }
      //Image3DType::Pointer referenceImage = referenceReader->GetOutput();
      //referenceImage->SetDirection(direction);
      //referenceImages.push_back(referenceImage);
  }
  //referenceReader->SetFileName(referencePath);
  //return EXIT_SUCCESS;

  //Image3DType::Pointer referenceImage = referenceReader->GetOutput();
  //Image3DType::DirectionType direction; direction.SetIdentity();
  //referenceImage->SetDirection(direction);
    
  
  /** Setup metamorphosis **/
  if(firstReferenceImage->GetLargestPossibleRegion().GetSize()[numberOfDimensions-1] <= 1) // If reference image is 2D...
  {
    // Do 2D metamorphosis
    Image3DType::RegionType region = firstReferenceImage->GetLargestPossibleRegion();
    Image3DType::SizeType   size = region.GetSize();
    size[numberOfDimensions-1] = 0;
    region.SetSize(size);

    typedef itk::ExtractImageFilter<Image3DType, Image2DType> ExtractorType;
    ExtractorType::Pointer extractor = ExtractorType::New();
    extractor->SetInput(firstReferenceImage);
    extractor->SetExtractionRegion(region);
    extractor->SetDirectionCollapseToIdentity();
    extractor->Update();

    return Metamorphosis<Image2DType>(extractor->GetOutput(), parser); 
  }
  else
  {
    // Do 3D metamorphosis
    return Metamorphosis<Image3DType>(firstReferenceImage, parser);
  }

} // end main


template<typename TImage>
int Metamorphosis(typename TImage::Pointer fixedImage, typename ParserType::Pointer parser)
{
  // Construct metamorphosis
  typedef TImage ImageType;
  typedef itk::MetamorphosisImageRegistrationMethodv4<ImageType,ImageType>  MetamorphosisType;
  typename MetamorphosisType::Pointer metamorphosis = MetamorphosisType::New();
  
  std::vector<std::string> referencePaths;
  parser->GetCommandLineArgument("--ref",referencePaths);
  unsigned int numImages = referencePaths.size();
  metamorphosis->SetChannels(numImages);
  
  // Read reference images
  std::vector<typename ImageType::Pointer> referenceImages;
  typedef itk::ImageFileReader<ImageType> ReaderType;
  typename ReaderType::Pointer referenceReader = ReaderType::New();
  typename ImageType::DirectionType direction; direction.SetIdentity();
  for(std::vector<std::string>::const_iterator i = referencePaths.begin(); i != referencePaths.end(); ++i)
  {
      //std::cout<<*i<<std::endl;
      referenceReader->SetFileName(*i);
      try
      {
          referenceReader->Update();
      }
      catch(itk::ExceptionObject& exceptionObject)
      {
          cerr<<"Error: Could not read reference image: "<<*i<<endl;
          cerr<<exceptionObject<<endl;
          return EXIT_FAILURE;
      }
      typename ImageType::Pointer referenceImage = referenceReader->GetOutput();
      //ImageType::DirectionType direction; direction.SetIdentity();
      referenceImage->SetDirection(direction);
      referenceImages.push_back(referenceImage);
  }
  //std::cout<<"length of referenceImages: "<<referenceImages.size()<<std::endl;
  metamorphosis->SetFixedImages(referenceImages);

  // Read input (moving) images
  std::vector<std::string> inputPaths;
  parser->GetCommandLineArgument("--in",inputPaths);
  if(inputPaths.size() != referencePaths.size())
  {
      cerr<<"Error: Number of input and reference images do not match: "<<endl;
      return EXIT_FAILURE;
  }

  std::vector<typename ImageType::Pointer> inputImages;
  typename ReaderType::Pointer inputReader = ReaderType::New();
  for(std::vector<std::string>::const_iterator i = inputPaths.begin(); i != inputPaths.end(); ++i)
  {
      //std::cout<<*i<<std::endl;
      inputReader->SetFileName(*i);
      //std::cout<<"set file name"<<std::endl;
      try
      {
          inputReader->Update();
      }
      catch(itk::ExceptionObject& exceptionObject)
      {
          cerr<<"Error: Could not read input image: "<<*i<<endl;
          cerr<<exceptionObject<<endl;
          return EXIT_FAILURE;
      }
      //std::cout<<"begin check for first interation"<<std::endl;
      if(i==inputPaths.begin())
      {
          //std::cout<<"first input iteration"<<std::endl;
          typename ImageType::Pointer firstInputImage = inputReader->GetOutput();
          firstInputImage->SetDirection(direction);
          metamorphosis->SetMovingImage(firstInputImage);
      }
      //std::cout<<"end of first iteration if statement"<<std::endl;
      typename ImageType::Pointer inputImage = inputReader->GetOutput();
      //ImageType::DirectionType direction; direction.SetIdentity();
      inputImage->SetDirection(direction);
      inputImages.push_back(inputImage);
      //std::cout<<"end of pushing back input images"<<std::endl;
  }
  //std::cout<<"length of inputImages: "<<inputImages.size()<<std::endl;
  metamorphosis->SetMovingImages(inputImages);
  
  // Read input (moving) mask
  itkStaticConstMacro(ImageDimension, unsigned int, ImageType::ImageDimension);
  typedef itk::ImageMaskSpatialObject<ImageDimension>  MaskType;
  typedef typename MaskType::ImageType                 MaskImageType;
  typedef itk::ImageFileReader<MaskImageType>          MaskReaderType;
  typename MaskImageType::Pointer inputMaskImage;
  typename MaskType::Pointer movingMask;

  string inputMaskPath;
  parser->GetCommandLineArgument("--inmask", inputMaskPath);

  if(inputMaskPath != "")
  {
    typename MaskReaderType::Pointer inputMaskReader = MaskReaderType::New();
    inputMaskReader->SetFileName(inputMaskPath);
    try
    {
      inputMaskReader->Update();
    }
    catch(itk::ExceptionObject& exceptionObject)
    {
      cerr<<"Error: Could not read input mask image: "<<inputMaskPath<<endl;
      cerr<<exceptionObject<<endl;
      return EXIT_FAILURE;
    }
    inputMaskImage = inputMaskReader->GetOutput();
    movingMask = MaskType::New();
    movingMask->SetImage(inputMaskImage);

  }

  // Set reference (fixed) image
  metamorphosis->SetFixedImage(fixedImage);   // I_1

  // Read reference (fixed) mask
  typename MaskType::Pointer fixedMask;
  string referenceMaskPath;
  parser->GetCommandLineArgument("--refmask", referenceMaskPath);

  if(referenceMaskPath != "")
  {
    typename MaskReaderType::Pointer referenceMaskReader = MaskReaderType::New();
    referenceMaskReader->SetFileName(referenceMaskPath);
    try
    {
      referenceMaskReader->Update();
    }
    catch(itk::ExceptionObject& exceptionObject)
    {
      cerr<<"Error: Could not read reference mask image: "<<referenceMaskPath<<endl;
      cerr<<exceptionObject<<endl;
      return EXIT_FAILURE;
    }

    fixedMask = MaskType::New();
    fixedMask->SetImage(referenceMaskReader->GetOutput());
  }

  // Set metamorphosis parameters 
  if(parser->ArgumentExists("--scale"))
  {
    float scale;
    parser->GetCommandLineArgument("--scale",scale);
    metamorphosis->SetScale(scale);
  }

  if(parser->ArgumentExists("--alpha"))
  {
    double alpha;
    parser->GetCommandLineArgument("--alpha",alpha);
    metamorphosis->SetRegistrationSmoothness(alpha);
  }


  if(parser->ArgumentExists("--beta"))
  {
    double beta;
    parser->GetCommandLineArgument("--beta",beta);

    if(beta < 0)
    {
      metamorphosis->UseBiasOff();
    }
    else
    {
      metamorphosis->UseBiasOn();
      metamorphosis->SetBiasSmoothness(beta);
    }
  }

  if(parser->ArgumentExists("--mu"))
  {
    double mu;
    parser->GetCommandLineArgument("--mu",mu);
    metamorphosis->SetMu(mu);
  }

  if(parser->ArgumentExists("--epsilon"))
  {
    double learningRate;
    parser->GetCommandLineArgument("--epsilon", learningRate);
    metamorphosis->SetLearningRate(learningRate);
  }

  if(parser->ArgumentExists("--epsilonmin"))
  {
    double minLearningRate;
    parser->GetCommandLineArgument("--epsilonmin", minLearningRate);
    metamorphosis->SetMinLearningRate(minLearningRate);
  }

  if(parser->ArgumentExists("--fraction"))
  {
    double minImageEnergyFraction;
    parser->GetCommandLineArgument("--fraction",minImageEnergyFraction);
    metamorphosis->SetMinImageEnergyFraction(minImageEnergyFraction);
  }

  if(parser->ArgumentExists("--steps"))
  {
    unsigned int numberOfTimeSteps;
    parser->GetCommandLineArgument("--steps",numberOfTimeSteps);
    metamorphosis->SetNumberOfTimeSteps(numberOfTimeSteps);
  }

  if(parser->ArgumentExists("--iterations"))
  {
    unsigned int numberOfIterations;
    parser->GetCommandLineArgument("--iterations",numberOfIterations);
    metamorphosis->SetNumberOfIterations(numberOfIterations);
  }


  if(parser->ArgumentExists("--verbose"))
  {
    typedef MetamorphosisObserver<MetamorphosisType>  MetamorphosisObserverType;
    typename MetamorphosisObserverType::Pointer observer = MetamorphosisObserverType::New();
    metamorphosis->AddObserver(itk::IterationEvent(),observer);
  }

  unsigned int numBins = 128;
  parser->GetCommandLineArgument("--bins", numBins);

  std::vector<unsigned int> costFunction;
  if(parser->ArgumentExists("--cost"))
  {
    //unsigned int costFunction;
    parser->GetCommandLineArgument("--cost", costFunction);

    switch(costFunction[0])
    {
      case 1:
      {
        typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType, ImageType> MetricType;
        typename MetricType::Pointer metric = MetricType::New();
        metric->SetNumberOfHistogramBins(numBins);
        metric->SetFixedImageMask(fixedMask);
        metric->SetMovingImageMask(movingMask);

        metamorphosis->SetMetric(metric);
        //TODO: get rid of this stuff and possibly need to set a vector instead of a float
        std::vector<double> mysigma;
        mysigma.push_back(0.0001);
        metamorphosis->SetSigma(mysigma);
        //metamorphosis->SetSigma2(0.0001);
	
        //typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType, ImageType> Metric2Type;
        //typename Metric2Type::Pointer metric2 = Metric2Type::New();
        //metric2->SetNumberOfHistogramBins(4);
        //metric2->SetFixedImageMask(fixedMask);
        //metric2->SetMovingImageMask(movingMask);
        //metamorphosis->SetMetric2(metric2);
        //metamorphosis->SetSigma(0.0001);
        //metamorphosis->SetSigma2(0.0001);

        break;
      }
      default:
      {
        typedef itk::MeanSquaresImageToImageMetricv4<ImageType, ImageType> MetricType;
        typename MetricType::Pointer metric = MetricType::New();
        metric->SetFixedImageMask(fixedMask);
        metric->SetMovingImageMask(movingMask);

        metamorphosis->SetMetric(metric);
        std::vector<double> mysigma;
        mysigma.push_back(1.0);
        metamorphosis->SetSigma(mysigma);
        //metamorphosis->SetSigma2(1.0);

        //typedef itk::MeanSquaresImageToImageMetricv4<ImageType, ImageType> Metric2Type;
        //typename Metric2Type::Pointer metric2 = Metric2Type::New();
        //metric2->SetFixedImageMask(fixedMask);
        //metric2->SetMovingImageMask(movingMask);
        //metamorphosis->SetMetric2(metric2);
        //metamorphosis->SetSigma(1.0);
        //metamorphosis->SetSigma2(1.0);

      }
    }

    typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType, ImageType> MIMetricType;
    typedef itk::MeanSquaresImageToImageMetricv4<ImageType, ImageType> MSEMetricType;
    typedef itk::ImageToImageMetricv4<ImageType, ImageType> ImageMetricType;
    // make a list of the base class for the metrics (ImagetoImageMetricv4 or something like that), then dynamic cast the metric inside the .hxx
    std::vector<typename ImageMetricType::Pointer> metrics;
    //std::vector<typename MIMetricType::Pointer> metrics;
    
    for(std::vector<unsigned int>::const_iterator i = costFunction.begin(); i != costFunction.end(); ++i)
    {
      //std::cout<<*i<<std::endl;
      switch(*i)
      {
          case 1:
          {
              typename MIMetricType::Pointer metric = MIMetricType::New();
              metric->SetNumberOfHistogramBins(numBins);
              metric->SetFixedImageMask(fixedMask);
              metric->SetMovingImageMask(movingMask);
              typename ImageMetricType::Pointer parentMetric = dynamic_cast<ImageMetricType *>(metric.GetPointer()); 
              metrics.push_back(parentMetric);
              break;
          }
          default:
          {
              typename MSEMetricType::Pointer metric = MSEMetricType::New();
              metric->SetFixedImageMask(fixedMask);
              metric->SetMovingImageMask(movingMask);
              typename ImageMetricType::Pointer parentMetric = dynamic_cast<ImageMetricType *>(metric.GetPointer()); 
              metrics.push_back(parentMetric);
          }
      }
          
    }
    metamorphosis->SetMetrics(metrics);
    
  }

  if(parser->ArgumentExists("--sigma"))
  {
    std::vector<double> sigma;
    parser->GetCommandLineArgument("--sigma",sigma);
    metamorphosis->SetSigma(sigma);
  }

  // Run metamorphosis 
  itk::TimeProbe clock;
  clock.Start();
  try
  {
    metamorphosis->Update();
  }
  catch(itk::ExceptionObject& exceptionObject)
  {
    cerr<<"Error: Metamorphosis did not terminate normally."<<endl;
    cerr<<exceptionObject<<endl;
    return EXIT_FAILURE;
  }
  clock.Stop();

  cout<<"E = "<<metamorphosis->GetEnergy()<<" ("<<metamorphosis->GetImageEnergyFraction()*100<<"%)"<<endl;
  cout<<"Length = "<<metamorphosis->GetLength()<<endl;
  cout<<"Time = "<<clock.GetTotal()<<"s"<<" ("<<clock.GetTotal()/60<<"m)"<<endl;

  // Write output images 
  int returnValue = EXIT_SUCCESS;

  // Compute I_0 o \phi_{10}
  typedef typename MetamorphosisType::OutputTransformType TransformType;
  typename TransformType::Pointer transform = const_cast<TransformType*>(metamorphosis->GetOutput()->Get()); // \phi_{10}
  transform->SetNumberOfIntegrationSteps(metamorphosis->GetNumberOfTimeSteps());
  transform->SetLowerTimeBound(1.0);
  transform->SetUpperTimeBound(0.0);
  transform->IntegrateVelocityField();

  typedef itk::WrapExtrapolateImageFunction<ImageType, double>         ExtrapolatorType;
  typedef typename TransformType::ScalarType ScalarType;
  typedef itk::ResampleImageFilter<ImageType, ImageType, ScalarType>   OutputResamplerType;
  typename OutputResamplerType::Pointer outputResampler = OutputResamplerType::New();
  
  // begin for loop to compute final transformed images
  // for now, just output first image
  
  // caster for moving image
  typedef itk::CastImageFilter<ImageType, ImageType> MovingCasterType;
  typename MovingCasterType::Pointer movingCaster = MovingCasterType::New();
  movingCaster->SetInput(inputImages[0]); 
  movingCaster->Update();
  typename ImageType::Pointer movingImage = movingCaster->GetOutput();

  outputResampler->SetInput(movingImage);   // I_0
  outputResampler->SetTransform(transform); // \phi_{10}
  outputResampler->UseReferenceImageOn();
  outputResampler->SetReferenceImage(fixedImage);
  outputResampler->SetExtrapolator(ExtrapolatorType::New());
  outputResampler->Update();

  // Compute I(1) = I_0 o \phi_{10} + B(1)
  typedef itk::AddImageFilter<ImageType,typename MetamorphosisType::BiasImageType,ImageType>  AdderType;
  typename AdderType::Pointer adder = AdderType::New();
  adder->SetInput1(outputResampler->GetOutput());  // I_0 o \phi_{10}
  adder->SetInput2(metamorphosis->GetBias());      // B(1)
  adder->Update();

  // Limit intensity of I(1) to intensity range of ouput image type
  typedef unsigned char   OutputPixelType;
  typedef itk::Image<OutputPixelType,ImageDimension>  OutputImageType;

  typedef itk::ClampImageFilter<ImageType, OutputImageType> ClamperType;
  typename ClamperType::Pointer clamper = ClamperType::New();
  clamper->SetInput(adder->GetOutput());
  clamper->SetBounds(itk::NumericTraits<OutputPixelType>::min(), itk::NumericTraits<OutputPixelType>::max());
  clamper->Update();

  typename OutputImageType::Pointer outputImage = clamper->GetOutput();

  // Write output image, I(1)
  string outputPath;
  parser->GetCommandLineArgument("--out",outputPath);

  typedef itk::ImageFileWriter<OutputImageType>  OutputWriterType;
  typename OutputWriterType::Pointer outputWriter = OutputWriterType::New();
  outputWriter->SetInput(outputImage); // I(1)
  outputWriter->SetFileName(outputPath);
  try
  {
    outputWriter->Update();
  }
  catch(itk::ExceptionObject& exceptionObject)
  {
    cerr<<"Error: Could not write output image: "<<outputPath<<endl;
    cerr<<exceptionObject<<endl;
    returnValue = EXIT_FAILURE;
  }

  // Write output mask
  string outMaskPath;
  parser->GetCommandLineArgument("--outmask", outMaskPath);
  
  if(outMaskPath != "" && inputMaskImage)
  {

    typedef itk::NearestNeighborInterpolateImageFunction<MaskImageType, ScalarType> MaskInterpolatorType;
    typename MaskInterpolatorType::Pointer maskInterpolator = MaskInterpolatorType::New();

    typedef itk::WrapExtrapolateImageFunction<MaskImageType, ScalarType> MaskExtrapolatorType;
    typename MaskExtrapolatorType::Pointer maskExtrapolator = MaskExtrapolatorType::New();    

    typedef itk::ResampleImageFilter<MaskImageType,MaskImageType, ScalarType>   MaskResamplerType;
    typename MaskResamplerType::Pointer maskResampler = MaskResamplerType::New();
    maskResampler->SetInput(inputMaskImage);
    maskResampler->SetTransform(transform); // phi_{10}
    maskResampler->UseReferenceImageOn();
    maskResampler->SetReferenceImage(fixedImage);
    maskResampler->SetInterpolator(maskInterpolator);
    maskResampler->SetExtrapolator(maskExtrapolator);

    // Write mask to file
    typename OutputWriterType::Pointer maskWriter = OutputWriterType::New();
    maskWriter->SetInput(maskResampler->GetOutput());
    maskWriter->SetFileName(outMaskPath);
    try
    {
      maskWriter->Update();
    }
    catch(itk::ExceptionObject& exceptionObject)
    {
      cerr<<"Error: Could not write mask image: "<<outMaskPath<<endl;
      cerr<<exceptionObject<<endl;
      returnValue = EXIT_FAILURE;
    }
  }

  // Write checker board of reference and output image
  string checkerPath;
  parser->GetCommandLineArgument("--checker", checkerPath);
  if(checkerPath != "")
  {
    typedef itk::CastImageFilter<ImageType, OutputImageType> OutputCasterType;
    typename OutputCasterType::Pointer fixedCaster = OutputCasterType::New();
    fixedCaster->SetInput(fixedImage);

    typedef itk::CheckerBoardImageFilter<OutputImageType> CheckerFilterType;
    typename CheckerFilterType::PatternArrayType pattern;
    pattern.Fill(4);

    typename CheckerFilterType::Pointer checker = CheckerFilterType::New();
    checker->SetInput1(fixedCaster->GetOutput());
    checker->SetInput2(outputImage);
    checker->SetCheckerPattern(pattern);
  
    typename OutputWriterType::Pointer checkerWriter = OutputWriterType::New();
    checkerWriter->SetInput(checker->GetOutput());
    checkerWriter->SetFileName(checkerPath);

    try
    {
      checkerWriter->Update();
    }
    catch(itk::ExceptionObject& exceptionObject)
    {
      cerr<<"Error: Could not write checker board image: "<<checkerPath<<endl;
      cerr<<exceptionObject<<endl;
      returnValue = EXIT_FAILURE;
    }
  }

  // Write displacement field, \phi_{10}
  typedef typename TransformType::DisplacementFieldType  FieldType;
  typedef itk::ImageFileWriter<FieldType>    FieldWriterType;

  string fieldPath;
  parser->GetCommandLineArgument("--field",fieldPath);

  if(fieldPath != "")
  {
    typename FieldWriterType::Pointer fieldWriter = FieldWriterType::New();
    fieldWriter->SetInput(transform->GetDisplacementField()); // \phi_{10}
    fieldWriter->SetFileName(fieldPath);
    try
    {
      fieldWriter->Update();
    }
    catch(itk::ExceptionObject& exceptionObject)
    {
      cerr<<"Error: Could not write displacement field: "<<fieldPath<<endl;
      cerr<<exceptionObject<<endl;
      returnValue = EXIT_FAILURE;
    }
  }

  // Write inverse displacement field, \phi_{01}
  string inverseFieldPath;
  parser->GetCommandLineArgument("--invfield", inverseFieldPath);
  if(inverseFieldPath != "")
  {
    typename FieldWriterType::Pointer inverseFieldWriter = FieldWriterType::New();
    inverseFieldWriter->SetInput(transform->GetInverseDisplacementField()); // \phi_{01}
    inverseFieldWriter->SetFileName(inverseFieldPath);
    try
    {
      inverseFieldWriter->Update();
    }
    catch(itk::ExceptionObject& exceptionObject)
    {
      cerr<<"Error: Could not write inverse displacement field: "<<inverseFieldPath<<endl;
      cerr<<exceptionObject<<endl;
      returnValue = EXIT_FAILURE;
    }
  }

  // Write bias, B(1)
  string biasPath;
  parser->GetCommandLineArgument("--bias",biasPath);

  if(biasPath != "")
  {
    typedef float                                       FloatPixelType;
    typedef itk::Image<FloatPixelType,ImageDimension>   FloatImageType;

    typedef itk::CastImageFilter<typename MetamorphosisType::BiasImageType,FloatImageType>  CasterType;
    typename CasterType::Pointer caster = CasterType::New();
    caster->SetInput(metamorphosis->GetBias()); // B(1)

    typedef itk::ImageFileWriter<FloatImageType>  BiasWriterType;
    typename BiasWriterType::Pointer biasWriter = BiasWriterType::New();
    biasWriter->SetInput(caster->GetOutput());
    biasWriter->SetFileName(biasPath);
    try
    {
      biasWriter->Update();
    }
    catch(itk::ExceptionObject& exceptionObject)
    {
      cerr<<"Error: Could not write bias image: "<<biasPath<<endl;
      cerr<<exceptionObject<<endl;
      returnValue = EXIT_FAILURE;
    }
  }
  

  // Write grid
  string gridPath;
  parser->GetCommandLineArgument("--grid", gridPath);

  if(gridPath != "")
  {
    std::vector<unsigned int> gridStep;
    parser->GetCommandLineArgument("--gridstep",gridStep);

    if(gridStep.size() == 1)
    {
      gridStep.resize(ImageDimension, gridStep[0]);
    }
    else if(gridStep.size() == 0 || gridStep.size() != ImageDimension)
    {
      gridStep.resize(ImageDimension, 5); // Default space in voxels between grid lines
    }

    // Generate grid
    typedef itk::BSplineKernelFunction<0>  KernelType;
    typename KernelType::Pointer kernelFunction = KernelType::New();
    typedef itk::GridImageSource<ImageType> GridSourceType;
    typename GridSourceType::Pointer gridSource = GridSourceType::New();
    typename GridSourceType::ArrayType gridSpacing;
    for(unsigned int i = 0; i < ImageDimension; i++){ gridSpacing[i] = fixedImage->GetSpacing()[i]*gridStep[i]; }
    //typename GridSourceType::ArrayType gridSpacing = fixedImage->GetSpacing()*gridStep;
    typename GridSourceType::ArrayType gridOffset; gridOffset.Fill(0.0);
    typename GridSourceType::ArrayType sigma = fixedImage->GetSpacing();
    typename GridSourceType::ArrayType which; which.Fill(true);

    gridSource->SetKernelFunction(kernelFunction);
    gridSource->SetSpacing(movingImage->GetSpacing());
    gridSource->SetOrigin(movingImage->GetOrigin());
    gridSource->SetSize(movingImage->GetLargestPossibleRegion().GetSize());
    gridSource->SetGridSpacing(gridSpacing);
    gridSource->SetGridOffset(gridOffset);
    gridSource->SetWhichDimensions(which);
    gridSource->SetSigma(sigma);
    gridSource->SetScale(itk::NumericTraits<unsigned char>::max());

    // Apply transform to grid
    typedef itk::WrapExtrapolateImageFunction<ImageType, ScalarType> ExtrapolatorType;
    typename ExtrapolatorType::Pointer extrapolator = ExtrapolatorType::New();    

    typedef itk::ResampleImageFilter<ImageType,OutputImageType, ScalarType>   GridResamplerType;
    typename GridResamplerType::Pointer gridResampler = GridResamplerType::New();
    gridResampler->SetInput(gridSource->GetOutput());
    gridResampler->SetTransform(transform); // phi_{10}
    gridResampler->UseReferenceImageOn();
    gridResampler->SetReferenceImage(fixedImage);
    gridResampler->SetExtrapolator(extrapolator);

    // Write grid to file
    typename OutputWriterType::Pointer gridWriter = OutputWriterType::New();
    gridWriter->SetInput(gridResampler->GetOutput());
    gridWriter->SetFileName(gridPath);
    try
    {
      gridWriter->Update();
    }
    catch(itk::ExceptionObject& exceptionObject)
    {
      cerr<<"Error: Could not write grid image: "<<gridPath<<endl;
      cerr<<exceptionObject<<endl;
      returnValue = EXIT_FAILURE;
    }
  }

  return returnValue;

} // end Metamorphosis


template<typename TMetamorphosis>
class MetamorphosisObserver: public itk::Command
{
public:
  /** Standard class typedefs. */
  typedef MetamorphosisObserver          Self;
  typedef itk::Command                   Superclass;
  typedef itk::SmartPointer<Self>        Pointer;
  typedef itk::SmartPointer<const Self>  ConstPointer;

  /** Method for creation throught object factory */
  itkNewMacro(Self);

  /** Filter typedefs */
  typedef TMetamorphosis*             FilterPointer;
  typedef const TMetamorphosis*       FilterConstPointer;
 
  /** Execute for non-const caller */
  void Execute(itk::Object *caller, const itk::EventObject &event)
  {
    FilterPointer filter = dynamic_cast<FilterPointer>(caller);
    ostringstream ss;

    if(itk::IterationEvent().CheckEvent(&event) )
    {
      if(filter->GetCurrentIteration() % 20 == 0) // Print column headings every 20 iterations
      {
        ss<<"\tE, E_velocity, E_rate, E_image (E_image %), LearningRate"<<std::endl;
      }
      //ss<<std::setprecision(4);// << std::fixed;
      ss<<filter->GetCurrentIteration()<<".\t"<<filter->GetEnergy()<<", "<<filter->GetVelocityEnergy()<<", "<<filter->GetRateEnergy()<<", "<<filter->GetImageEnergy()<<" ("<<filter->GetImageEnergyFraction()*100<<"%), ";
      ss.setf(std::ios::scientific,std::ios::floatfield);
      ss<<filter->GetLearningRate()<<std::endl;
      std::cout<<ss.str();
    }    
  }

  /** Execute for non-const caller */
  void Execute(const itk::Object* caller, const itk::EventObject &event)
  {
    FilterConstPointer filter = dynamic_cast<FilterConstPointer>(caller);
  }

protected:
  MetamorphosisObserver(){}; // Constructor
};  // end class MetamorphosisObserver
