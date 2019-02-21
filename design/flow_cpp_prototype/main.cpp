// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>

#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"

using namespace std;
using namespace cv;

int w = 1280; //320 //1280
int h = 720; //240 //720
int ww = 320;//640; //160
int hh = 240;//480; //120

static Mat getVisibleFlow(InputArray flow)
{
    vector<UMat> flow_vec;
    split(flow, flow_vec);
    UMat magnitude, angle;
    cartToPolar(flow_vec[0], flow_vec[1], magnitude, angle, true);
    magnitude.convertTo(magnitude, CV_32F, 0.5);
    vector<UMat> hsv_vec;
    hsv_vec.push_back(angle);
    hsv_vec.push_back(UMat::ones(angle.size(), angle.type()));
    hsv_vec.push_back(magnitude);
    UMat hsv;
    merge(hsv_vec, hsv);
    Mat img;
    cvtColor(hsv, img, COLOR_HSV2BGR);
    return img;
}

static Mat flow_directions(InputArray flow) {
    vector<UMat> flow_vec;
    split(flow, flow_vec);
    UMat magnitude, angle;
    cartToPolar(flow_vec[0], flow_vec[1], magnitude, angle, true);
    magnitude.convertTo(magnitude, CV_32F, 0.5);

    Mat filtered;
    threshold(magnitude, filtered, 0.3, 1, THRESH_BINARY);
    Mat filtered8u;
    filtered.convertTo(filtered8u, CV_8U);

    UMat angleFiltered;
//    angle.copyTo(angleFiltered, filtered);
    bitwise_and(angle, angle, angleFiltered, filtered8u);

    UMat ufiltered;
    filtered.copyTo(ufiltered);

    UMat up;
    inRange(angleFiltered, 45, 135, up);
    int upMag = countNonZero(up);

    UMat left;
    inRange(angleFiltered, 135, 225, left);
    int leftMag = countNonZero(left);

    UMat down;
    inRange(angleFiltered, 225, 315, down);
    int downMag = countNonZero(down);

//    int rightMag = 0;
    UMat right1;
    inRange(angleFiltered, 0.1, 45, right1);
    int rightMag = countNonZero(right1);
    UMat right2;
    inRange(angleFiltered, 315, 359.5, right2);
    rightMag = rightMag + countNonZero(right2);



//    vector<UMat> up_vec;
//    split(up, up_vec);


    vector<UMat> hsv_vec;
    hsv_vec.push_back(angle);
    hsv_vec.push_back(UMat::ones(angle.size(), angle.type()));
    hsv_vec.push_back(ufiltered);
    UMat hsv;
    merge(hsv_vec, hsv);
    Mat img;
    cvtColor(hsv, img, COLOR_HSV2BGR);

    ostringstream buf;
    buf << rightMag << " " << upMag << " " << leftMag << " " << downMag;
    putText(img, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255), 2, LINE_AA);

    return img;
}


static Size fitSize(const Size & sz,  const Size & bounds)
{
//    CV_Assert(!sz.empty());
    if (sz.width > bounds.width || sz.height > bounds.height)
    {
        double scale = std::min((double)bounds.width / sz.width, (double)bounds.height / sz.height);
        return Size(cvRound(sz.width * scale), cvRound(sz.height * scale));
    }
    return sz;
}

int main(int argc, const char* argv[])
{
    const char* keys =
            "{ h help     |     | print help message }"
            "{ c camera   | 0   | capture video from camera (device index starting from 0) }"
            "{ a algorithm | fb | algorithm (supported: 'fb', 'dis')}"
            "{ m cpu      |     | run without OpenCL }"
            "{ v video    |     | use video as input }"
            "{ o original |     | use original frame size (do not resize to 640x480)}"
    ;
    CommandLineParser parser(argc, argv, keys);
    parser.about("This sample demonstrates using of dense optical flow algorithms.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    int camera = parser.get<int>("camera");
    string algorithm = parser.get<string>("algorithm");
    bool useCPU = parser.has("cpu");
    string filename = parser.get<string>("video");
    bool useOriginalSize = parser.has("original");
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    VideoCapture cap;
    if(filename.empty())
        cap.open(camera);
    else
        cap.open(filename);
    if (!cap.isOpened())
    {
        cout << "Can not open video stream: '" << (filename.empty() ? "<camera>" : filename) << "'" << endl;
        return 2;
    }
//    cap.set(CAP_PROP_FRAME_WIDTH,640);
//    cap.set(CAP_PROP_FRAME_HEIGHT,480);
    cap.set(CAP_PROP_FRAME_WIDTH, w);
    cap.set(CAP_PROP_FRAME_HEIGHT, h);
    std::this_thread::sleep_for(1s);

//    Ptr<DenseOpticalFlow> alg;
    cv::Ptr<cv::DenseOpticalFlow> alg;
    if (algorithm == "fb")
        alg = FarnebackOpticalFlow::create();
    else if (algorithm == "dis")
        alg = cv::FarnebackOpticalFlow::create();
//        alg = DISOpticalFlow::create(DISOpticalFlow::PRESET_FAST);
    else
    {
        cout << "Invalid algorithm: " << algorithm << endl;
        return 3;
    }

    ocl::setUseOpenCL(!useCPU);

    cout << "Press 'm' to toggle CPU/GPU processing mode" << endl;
    cout << "Press ESC or 'q' to exit" << endl;

    UMat prevFrame, frame, input_frame, flow;
    for(;;)
    {
        if (!cap.read(input_frame) || input_frame.empty())
        {
            cout << "Finished reading: empty frame" << endl;
            break;
        }
//        Size small_size = fitSize(input_frame.size(), Size(480, 320));
//        Size small_size = fitSize(input_frame.size(), Size(320, 240));
        Size small_size = fitSize(input_frame.size(), Size(ww, hh));
        if (!useOriginalSize && small_size != input_frame.size())
            resize(input_frame, frame, small_size);
        else
            frame = input_frame;
        cvtColor(frame, frame, COLOR_BGR2GRAY);
        imshow("frame", frame);
        if (!prevFrame.empty())
        {
            int64 t = getTickCount();
            alg->calc(prevFrame, frame, flow);
            t = getTickCount() - t;
            {
                Mat img = getVisibleFlow(flow);
                ostringstream buf;
//                buf << "Algo: " << algorithm << " | "
//                    << "Mode: " << (useCPU ? "CPU" : "GPU") << " | "
                   buf << "FPS: " << fixed << setprecision(1) << (getTickFrequency() / (double)t);
                putText(img, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2, LINE_AA);
                imshow("Dense optical flow field", img);

                Mat fd = flow_directions(flow);
                imshow("FD", fd);
            }
        }
        frame.copyTo(prevFrame);

        // interact with user
        const char key = (char)waitKey(30);
        if (key == 27 || key == 'q') // ESC
        {
            cout << "Exit requested" << endl;
            break;
        }
        else if (key == 'm')
        {
            useCPU = !useCPU;
            ocl::setUseOpenCL(!useCPU);
            cout << "Set processing mode to: " << (useCPU ? "CPU" : "GPU") << endl;
        }
    }

    return 0;
}

int test()
{
    int W = 52;             // window size is WxW
    double C_Thr = 0.43;    // threshold for coherency
    int LowThr = 35;        // threshold1 for orientation, it ranges from 0 to 180
    int HighThr = 57;       // threshold2 for orientation, it ranges from 0 to 180

    Mat imgIn = imread("input.jpg", IMREAD_GRAYSCALE);
    if (imgIn.empty()) //check whether the image is loaded or not
    {
        cout << "ERROR : Image cannot be loaded..!!" << endl;
        return -1;
    }

    //! [main_extra]
    //! [main]
    Mat imgCoherency, imgOrientation;
    //calcGST(imgIn, imgCoherency, imgOrientation, W);

    //! [thresholding]
    Mat imgCoherencyBin;
    imgCoherencyBin = imgCoherency > C_Thr;

    return 0;
}