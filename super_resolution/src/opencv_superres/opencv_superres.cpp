#include <iostream>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/superres.hpp"
#include "opencv2/superres/optical_flow.hpp"
#include "opencv2/opencv_modules.hpp"

#include "opencv2/cudaarithm.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;

class ImageFrameSource : public FrameSource {
    Mat mFrame;

public:
    ImageFrameSource(Mat& frame) {
        frame.copyTo(mFrame);
    }

    void setFrame(Mat& frame) {
        frame.copyTo(mFrame);
    }

    virtual void nextFrame(OutputArray frame) {
        if (frame.isGpuMat()) {
            frame.getGpuMatRef().upload(mFrame);
        } else {
            mFrame.copyTo(frame);
        }
    }

    virtual void reset() {

    }
};

static Ptr<DenseOpticalFlowExt> createOptFlow(const string& name, bool useGpu) {
    if (name == "farneback") {
        if (useGpu)
            return createOptFlow_Farneback_CUDA();
        else
            return createOptFlow_Farneback();
    }

    if (name == "tvl1") {
        if (useGpu)
            return createOptFlow_DualTVL1_CUDA();
        else
            return createOptFlow_DualTVL1();
    }

    if (name == "brox") {
        if (useGpu)
            return createOptFlow_Brox_CUDA();
    }

    if (name == "pyrlk") {
        if (useGpu)
            return createOptFlow_PyrLK_CUDA();
    }

    cerr << "Incorrect Optical Flow algorithm - " << name << endl;

    return Ptr<DenseOpticalFlowExt>();
}

int main(int argc, const char* argv[]) {
    CommandLineParser cmd(argc, argv,
		"{ m image      |           | Input image (mandatory)}"
		"{ o output     |           | Output image }"
		"{ s scale      | 4         | Scale factor }"
		"{ i iterations | 180       | Iteration count }"
		"{ t temporal   | 4         | Radius of the temporal search area }"
		"{ f flow       | farneback | Optical flow algorithm (farneback, tvl1, brox, pyrlk) }"
		"{ g gpu        | false     | CPU as default device, cuda for CUDA }"
		"{ h help       | false     | Print help message }"
    );

    //
    // Parse arguments
    //

    //
    auto inputImageName = cmd.get<string>("image");
    auto outputImageName = cmd.get<string>("output");
    auto scale = cmd.get<int>("scale");
    auto iterations = cmd.get<int>("iterations");
    auto temporalAreaRadius = cmd.get<int>("temporal");
    auto optFlow = cmd.get<string>("flow");
    auto gpuOption = cmd.get<string>("gpu");

    //
    if (cmd.get<bool>("help") || inputImageName.empty()) {
        cout << "This sample demonstrates Super Resolution algorithms for image" << endl;
        cmd.printMessage();
        return EXIT_SUCCESS;
    }

    //
    // Prepare super resolution configurations
    //

    // Check to use CUDA
    transform(gpuOption.begin(), gpuOption.end(), gpuOption.begin(), ::tolower);

    bool useCuda = (gpuOption == "cuda");

    Ptr<SuperResolution> superRes;

    if (useCuda) {
        superRes = createSuperResolution_BTVL1_CUDA();
    } else {
        superRes = createSuperResolution_BTVL1();
    }

    // Check which optical flow algorithm to use
    Ptr<DenseOpticalFlowExt> of = createOptFlow(optFlow, useCuda);

    if (of.empty()) {
        return EXIT_FAILURE;
    }

    // Load input image
    Mat inputImage = imread(inputImageName);
    Mat lowResImage;
    resize(inputImage, lowResImage, cv::Size(0, 0), 0.5, 0.5, cv::INTER_CUBIC);
    Mat interImage;
    resize(lowResImage, interImage, cv::Size(0, 0), 2, 2, cv::INTER_CUBIC);
    ImageFrameSource imageFrameSource(lowResImage);
    Ptr<ImageFrameSource> frameSource = makePtr<ImageFrameSource>(imageFrameSource);

    // Set super resolver configurations
    superRes->setInput(frameSource);
    superRes->setScale(scale);
    superRes->setIterations(iterations);
    superRes->setTemporalAreaRadius(temporalAreaRadius);
    superRes->setOpticalFlow(of);

    // Print super resolver configurations
    cout << "Input           : " << inputImageName << " " << lowResImage.size() << endl;
    cout << "Scale factor    : " << scale << endl;
    cout << "Iterations      : " << iterations << endl;
    cout << "Temporal radius : " << temporalAreaRadius << endl;
    cout << "Optical Flow    : " << optFlow << endl;
    cout << "Mode            : " << (useCuda ? "CUDA" : "CPU") << endl;

    //
    // Super resolve the input image
    //

    // Start timer and begin the super resolution
    Mat result;
    TickMeter tm;
    tm.start();
    superRes->nextFrame(result);
    tm.stop();
    cout << "Finish in " << tm.getTimeSec() << " sec" << endl;

    // // Save images
    // system("pwd");
    // imwrite("data/ground_truth.png", inputImage);
    // imwrite("data/interpolation.png", interImage);
    // imwrite("data/super_resolution.png", result);

    // // Show the super resolved image
    // imshow("Ground Truth", inputImage);
    // imshow("Interpolation", interImage);
    // imshow("Super Resolution", result);
    // waitKey(0);

    return 0;
}
