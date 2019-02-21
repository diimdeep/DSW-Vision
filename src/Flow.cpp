#include "Vision.hpp"

// #include "widget/FramebufferWidget.hpp"
// #include <nanovg.h>
// #include <blendish.h>

#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>

#include "colors.h"

namespace dsw_vision {

using namespace std;
using namespace cv;
using std::cout; using std::cerr; using std::endl;
// fdfs
static Mat getVisibleFlow(InputArray flow)
{
    vector<UMat> flow_vec;
    split(flow, flow_vec);
    UMat magnitude, angle;
    cartToPolar(flow_vec[0], flow_vec[1], magnitude, angle, true);
    magnitude.convertTo(magnitude, CV_32F, 0.2);
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

static Size fitSize(const Size & sz,  const Size & bounds)
{
    CV_Assert(!sz.empty());
    if (sz.width > bounds.width || sz.height > bounds.height)
    {
        double scale = std::min((double)bounds.width / sz.width, (double)bounds.height / sz.height);
        return Size(cvRound(sz.width * scale), cvRound(sz.height * scale));
    }
    return sz;
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


// struct OpenGlWidget : FramebufferWidget {
// 	/** Draws every frame by default
// 	Override this and call `FramebufferWidget::step()` to restore the default behavior of FramebufferWidget.
// 	*/
// 	virtual void step() {
// 		// Render every frame
// 		dirty = true;
// 	}
// 	/** Draws to the framebuffer.
// 	Override to initialize, draw, and flush the OpenGL state.
// 	*/
// 	virtual void drawFramebuffer() {
// 		glViewport(0.0, 0.0, fbSize.x, fbSize.y);
// 		glClearColor(0.0, 0.0, 0.0, 1.0);
// 		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

// 		glMatrixMode(GL_PROJECTION);
// 		glLoadIdentity();
// 		glOrtho(0.0, fbSize.x, 0.0, fbSize.y, -1.0, 1.0);

// 		glBegin(GL_TRIANGLES);
// 		glColor3f(1, 0, 0);
// 		glVertex3f(0, 0, 0);
// 		glColor3f(0, 1, 0);
// 		glVertex3f(fbSize.x, 0, 0);
// 		glColor3f(0, 0, 1);
// 		glVertex3f(0, fbSize.y, 0);
// 		glEnd();
// 	}
// };

struct RenderData {
	uchar *image;
	int width;
	int height;
	std::atomic<bool> *dirty;
	std::atomic<bool> *free;
};

void * cameraOpenCVWorker(RenderData data) {
	std::cout << "Opening camera..." << std::endl;
	VideoCapture capture;
	capture.open(0);
    if (!capture.isOpened())
    {
        std::cerr << "ERROR: Can't initialize camera capture" << std::endl;
        return 0;
    }
    capture.set(CAP_PROP_FRAME_WIDTH, 320);
	capture.set(CAP_PROP_FRAME_HEIGHT, 240);
	std::chrono::seconds sec(1);
	this_thread::sleep_for(sec);

	// std::cout << "Creating algo..." << std::endl;
	Ptr<DenseOpticalFlow> alg = FarnebackOpticalFlow::create();
	// std::cout << "Enabling OpenCL" << std::endl;
	ocl::setUseOpenCL(true);

	// std::cout << "Capturing.." << std::endl;
	UMat prevFrame, frame, input_frame, flow;
	for(;;) {
		if (!capture.read(input_frame) || input_frame.empty())
        {
            cout << "Finished reading: empty frame" << endl;
            break;
        }

        // std::cout << "Resize.." << std::endl;
        Size small_size = fitSize(input_frame.size(), Size(160, 120));
        resize(input_frame, frame, small_size);

		cvtColor(frame, frame, COLOR_BGR2GRAY);

		if (!prevFrame.empty())
        {
        	std::cout << "Run algo.." << std::endl;
            int64 t = getTickCount();
            alg->calc(prevFrame, frame, flow);
            t = getTickCount() - t;
            {
                Mat img = flow_directions(flow);

                ostringstream buf;
                   buf << "FPS: " << fixed << setprecision(1) << (getTickFrequency() / (double)t);
                putText(img, buf.str(), Point(10, 60), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 2, LINE_AA);

                std::cout << "Gen bitmap.." << std::endl;
		        // cv::cvtColor(img, img, COLOR_BGR2BGRA, 4);
		        uchar* frameData = new uchar[img.total()*4];
		        Mat continuousRGBA(img.size(), CV_8UC4, frameData);
		        // Mat continuousRGBA(img.size(), CV_8UC4);
				cv::cvtColor(img, continuousRGBA, COLOR_BGR2RGBA, 4);

				std::cout << "Store bitmap.." << std::endl;
				data.free->store(false);
				data.image = frameData;
				// data.image = continuousRGBA.ptr();
				data.free->store(true);
				data.dirty->store(true);
				std::cout << "Stored." << std::endl;

            }
        }
        frame.copyTo(prevFrame);
	}

	// for (int i = 0; i < 60; ++i)
	// {
	// 	if (!capture.read(input_frame) || input_frame.empty())
	//     {
	//         cout << "Finished reading: empty frame" << endl;
	//     }
	// 	Size small_size = fitSize(input_frame.size(), Size(640, 480));
	// 	resize(input_frame, frame, small_size);
	//     cvtColor(frame, frame, COLOR_BGR2GRAY);

	//     // imshow("frame", frame);

	//     if(!prevFrame.empty()) {
	//     	alg->calc(prevFrame, frame, flow);
	//     	Mat img = getVisibleFlow(flow);
	//     	imshow("DOFF", img);
	//     }
	//     frame.copyTo(prevFrame);
	// }
	return 0;
}

struct Flow : Module {
	enum ParamIds {
		NUM_PARAMS
	};
	enum InputIds {
		NUM_INPUTS
	};
	enum OutputIds {
		LIGHT_OUTPUT,
		NUM_OUTPUTS
	};
	enum LightIds {
		BLINK_LIGHT,
		NUM_LIGHTS
	};
	std::atomic<bool> renderDataFree;
	std::atomic<bool> dirty;

	RenderData renderData;
	thread opencvThread;

	Flow() : Module(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS) {
		renderDataFree.store(true);
		dirty.store(false);
		renderData.free = &renderDataFree;
		renderData.dirty = &dirty;
		renderData.width = 160;
		renderData.height = 120;

		opencvThread = thread(cameraOpenCVWorker, std::ref(renderData));
		opencvThread.detach();

		// int taille = 500;
		// Mat image(taille,taille,CV_8UC4);
		// for(int y = 0; y < taille; y++){
		//    Vec3b val;
		//    val[0] = 0;
		//    val[1] = (y*255)/taille;
		//    val[2] = (taille-y)*255/taille;
		//    val[3] = 1;
		//    for(int x = 0; x < taille; x++)
		//       image.at<Vec3b>(y,x) = val;
		// }
		int taille = 500;
		Mat image(taille,taille,CV_8UC3);
		for(int y = 0; y < taille; y++){
		   Vec3b val;
		   val[0] = 0; val[1] = (y*255)/taille; val[2] = (taille-y)*255/taille;
		   for(int x = 0; x < taille; x++)
		      image.at<Vec3b>(y,x) = val;
		}
		imshow("image", image);
	}

	void step() override;
};

void Flow::step() {
	// unibi = (params[UNIBI_PARAM].value > 0.0f);
	outputs[LIGHT_OUTPUT].value = 1.0;

	while(!renderDataFree){
	}
}

struct RenderWidget : OpaqueWidget {
	Flow *module;
	const float width = 160.0f;
	const float height = 120.0f;
	int img = 0;
	uchar* preloadImage;
	// std::shared_ptr<char> raiiArray;
	vector<uchar> vect;

	RenderWidget(){
		std::cout << "Gen image.." << std::endl;
		int taille = 500;
		Mat image(taille,taille,CV_8UC3);
		for(int y = 0; y < taille; y++){
		   Vec3b val;
		   val[0] = 0; val[1] = (y*255)/taille; val[2] = (taille-y)*255/taille;
		   for(int x = 0; x < taille; x++)
		      image.at<Vec3b>(y,x) = val;
		}

		// cv::cvtColor(image, image, COLOR_BGR2RGBA, 4);
		int arrayLength = image.total()*4;

		vect = vector<uchar>(arrayLength);
		// raiiArray = std::shared_ptr<char>(new char[arrayLength], std::default_delete<char[]>());
		preloadImage = new uchar[arrayLength];
        Mat continuousRGBA(image.size(), CV_8UC4, vect.data());
        // Mat continuousRGBA(img.size(), CV_8UC4);
		cv::cvtColor(image, continuousRGBA, COLOR_BGR2RGBA, 4);
	}

	void draw(NVGcontext *vg) override {
		if(img == 0) {
			img = nvgCreateImageRGBA(vg, 500, 500, 0, vect.data());
		}


		if(module->renderDataFree && module->dirty) {
			std::cout << "Read bitmap.." << std::endl;
			// int nvgCreateImageRGBA(NVGcontext* ctx, int w, int h, int imageFlags, const unsigned char* data);
			// crashing
			// nvgUpdateImage(vg, img, module->renderData.image);
			nvgUpdateImage(vg, img, preloadImage);

			std::cout << "DidRead." << std::endl;
		}

		std::cout << "Paint.." << std::endl;
		nvgBeginPath(vg);
		// if (module->width>0 && module->height>0)
		// 	nvgScale(vg, width/module->width, height/module->height);
	 	NVGpaint imgPaint = nvgImagePattern(vg, 0, 0, module->renderData.width,module->renderData.height, 0, img, 1.0f);
	 	nvgRect(vg, 0, 0, module->renderData.width, module->renderData.height);
	 	nvgFillPaint(vg, imgPaint);
	 	nvgFill(vg);
		nvgClosePath(vg);
		std::cout << "DidPaint." << std::endl;
		module->dirty.store(false);

	}
};

struct FlowWidget : ModuleWidget {
	FlowWidget(Flow *module) : ModuleWidget(module) {
		setPanel(SVG::load(assetPlugin(plugin, "res/Flow.svg")));

		RenderWidget *display = new RenderWidget();
		display->module = module;
		display->box.pos = rack::Vec(9, 30);
		display->box.size = rack::Vec(160, 120);
		addChild(display);
	}
};

}

Model *modelFlowModule = Model::create<dsw_vision::Flow, dsw_vision::FlowWidget>("DSW-Vision", "Flow", "Flow", OSCILLATOR_TAG);
