#include "Vision.hpp"

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
#include "controls.hpp"

namespace dsw_vision {

using namespace std;
using namespace cv;
using std::cout; using std::cerr; using std::endl;

int W = 160;
int H = 120;

struct Directions {
	float right;
	float up;
	float left;
	float down;
	float angle;
};

struct RenderData {
	int 			  width;
	int 			  height;
	vector<uchar> 	  *image; 	// uchar *image;
	Directions 		  *directions;
	std::atomic<bool> *dirty;
	std::atomic<bool> *free;
	std::atomic<bool> *read;
	std::atomic<bool> *useCPUorOpenCL;
	std::atomic<float> *sens;
	std::atomic<int> *viewType;
};

static Mat getVisibleFlow(InputArray flow, float sens)
{
    vector<UMat> flow_vec;
    split(flow, flow_vec);
    UMat magnitude, angle;
    cartToPolar(flow_vec[0], flow_vec[1], magnitude, angle, true);
    magnitude.convertTo(magnitude, CV_32F, 0.5*sens);
    vector<UMat> hsv_vec;
    hsv_vec.push_back(angle);
    hsv_vec.push_back(UMat::ones(angle.size(), angle.type()));
    hsv_vec.push_back(magnitude);
    UMat hsv;
    merge(hsv_vec, hsv);
    Mat img;
    cvtColor(hsv, img, COLOR_HSV2BGR);
    Mat img8;
    img.convertTo(img8, CV_8UC3, 255);
    // flip(img8, img8, 1);
    return img8;
}


static Mat draw_directions_over_image(Directions directions, Mat img){
    ostringstream buf;
    buf << fixed << setprecision(1) << directions.up << " " << directions.down << " " << directions.left << " " << directions.right;
    putText(img, buf.str(), Point(1, 10), FONT_HERSHEY_PLAIN, 0.9, Scalar(255, 255, 255), 1, LINE_AA);
    return img;
}

static Mat draw_fps_over_image(float fps, Mat img){
    ostringstream buf;
	buf << "FPS: " << fixed << setprecision(0) << fps;
	putText(img, buf.str(), Point(1, 12), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, LINE_AA);
    return img;
}

static Mat draw_ms_over_image(float fps, Mat img){
    ostringstream buf;
	buf << "ms: " << fixed << setprecision(1) << fps;
	putText(img, buf.str(), Point(60, 12), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, LINE_AA);
    return img;
}

// int width = 160;
// int height = 120;
// Mat img = draw_gradient(width, height);
// imshow("Dense optical flow field", img);
static Mat draw_gradient(int width, int height) {
	Mat image(width, height, CV_8UC3);
	for(int y = 0; y < height; y++){
		Vec3b val;
		val[0] = 0;
		val[1] = (y*255)/height;
		val[2] = (height-y)*255/height;
		for(int x = 0; x < width; x++) {
			image.at<Vec3b>(y,x) = val;
		}
	}
	return image;
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


static Directions flow_directions(InputArray flow, float sens) {
    vector<UMat> flow_vec;
    split(flow, flow_vec);
    UMat magnitude, angle;
    cartToPolar(flow_vec[0], flow_vec[1], magnitude, angle, true);
    magnitude.convertTo(magnitude, CV_32F, 0.5*sens);

    Mat filtered;
    threshold(magnitude, filtered, 0.4, 1, THRESH_BINARY);
    Mat filtered8u;
    filtered.convertTo(filtered8u, CV_8U);

    UMat angleFiltered;
    bitwise_and(angle, angle, angleFiltered, filtered8u);

    UMat ufiltered;
    filtered.copyTo(ufiltered);

    float scale = 1000.0f;
    Directions d;
    UMat up;
    inRange(angleFiltered, 45, 135, up);
    d.down = countNonZero(up)/scale;

    UMat left;
    inRange(angleFiltered, 135, 225, left);
    d.left = countNonZero(left)/scale;

    UMat down;
    inRange(angleFiltered, 225, 315, down);
    d.up = countNonZero(down)/scale;

//    int rightMag = 0;
    UMat right1;
    inRange(angleFiltered, 0.1, 45, right1);
    d.right = countNonZero(right1);
    UMat right2;
    inRange(angleFiltered, 315, 359.9, right2);
    d.right = d.right + countNonZero(right2);
    d.right = d.right/scale;

    double mean = cv::mean(angleFiltered)[0];
    d.angle = mean;


    return d;
}

void * cameraOpenCVWorker(RenderData data) {
	std::cout << "Opening camera..." << std::endl;
	VideoCapture capture;
	capture.open(0);
    if (!capture.isOpened())
    {
        std::cerr << "ERROR: Can't initialize camera capture" << std::endl;
        return 0;
    }
    // capture.set(CAP_PROP_EXPOSURE, 0);
  	// capture.set(CAP_PROP_FPS, 30);
    capture.set(CAP_PROP_FRAME_WIDTH, 320);
	capture.set(CAP_PROP_FRAME_HEIGHT, 240);
	capture.set(CAP_PROP_CONVERT_RGB, false);
	std::chrono::milliseconds ms(500);
	this_thread::sleep_for(ms); // IDK BUT IT THROWS..

	// std::cout << "Enabling OpenCL" << std::endl;
	bool useCPUorOpenCL = data.useCPUorOpenCL->load();
	std::cout << "Changing option to use CPU(0)/GPU(1) = " << useCPUorOpenCL << std::endl;
	ocl::setUseOpenCL(useCPUorOpenCL);

	// std::cout << "Creating algo..." << std::endl;
	Ptr<DenseOpticalFlow> alg = FarnebackOpticalFlow::create();

	// std::cout << "Capturing.." << std::endl;
	UMat prevFrame, frame, frameGray, input_frame, flow;
	for(;;) {
		double startTime = getTickCount();

		bool useCPUorOpenCL_ = data.useCPUorOpenCL->load();
		if(useCPUorOpenCL != useCPUorOpenCL_){
			useCPUorOpenCL = useCPUorOpenCL_;
			std::cout << "Changing option to use CPU(0)/GPU(1) = " << useCPUorOpenCL << std::endl;
			ocl::setUseOpenCL(useCPUorOpenCL);
		}

		if (!capture.read(input_frame) || input_frame.empty())
        {
            cout << "Finished reading: empty frame" << endl;
            break;
        }

        // std::cout << "Resize.." << std::endl;
        Size small_size = fitSize(input_frame.size(), Size(W, H));
        resize(input_frame, frame, small_size);
        flip(frame, frame, 1);

		cvtColor(frame, frameGray, COLOR_BGR2GRAY);


		if (!prevFrame.empty())
        {
        	// std::cout << "Run algo.." << std::endl;

            alg->calc(prevFrame, frameGray, flow);

            // std::cout << "Gen bitmap.." << std::endl;

            float sens = data.sens->load();

			Mat img;
            int viewType = data.viewType->load();
            if (viewType == 0) {
        		img = getVisibleFlow(flow, sens);
            } else if (viewType == 1) {
            	cvtColor(frame, img, COLOR_RGB2RGBA);
            }

        	//CV_32FC3 isContinuous() 1
        	// cout << "getVisibleFlow format " << typeToString(img.type()) << " isContinuous() " << img.isContinuous() << endl;

			Directions directions = flow_directions(flow, sens);
        	// img = draw_directions_over_image(directions, img);

			double endTime = getTickCount();
			double frameTime = endTime - startTime;
			double freq = getTickFrequency();
			double ms = 1000.0*frameTime / freq;
        	double fps = (freq / (double)frameTime);
        	img = draw_fps_over_image(fps, img);
        	img = draw_ms_over_image(ms, img);

			// img = cv::imread(assetPlugin(plugin, "design/pic5160x120.png"), IMREAD_COLOR);//IMREAD_UNCHANGED);//
			// CV_8UC3 isContinuous() 1
			// cout << "Image format " << typeToString(img.type()) << " isContinuous() " << img.isContinuous() << endl;
	        // cv::cvtColor(img, img, COLOR_BGR2BGRA, 4);
	        int arrayLength = img.total()*4;

			// vector<uchar> frameData_ptr = vector<uchar>(arrayLength);
	        uchar* frameData = new uchar[arrayLength];
	        Mat continuousRGBA(img.size(), CV_8UC4, frameData);
	        // Mat continuousRGBA(img.size(), CV_8UC4, frameData.get()->data());
	        // Mat continuousRGBA(img.size(), CV_8UC4);
			// cv::cvtColor(img, continuousRGBA, COLOR_BGR2BGRA, 4);
			cv::cvtColor(img, continuousRGBA, COLOR_BGR2RGBA, 4);

			// std::cout << "Store bitmap.." << std::endl;
			data.free->store(false);
			memcpy(data.image->data(), frameData, arrayLength * sizeof(uchar));
			// data.image->data = frameData;
			data.directions->angle = directions.angle;
			data.directions->up = directions.up;
			data.directions->left = directions.left;
			data.directions->down = directions.down;
			data.directions->right = directions.right;
			data.free->store(true);
			data.dirty->store(true);
			data.read->store(true);

			// if (fps > 30) {
			// 	this_thread::sleep_for(chrono::duration<double, std::micro>(ms/1000.0));
			// }
        }
        frameGray.copyTo(prevFrame);

        while(data.read->load()){
		}
	}
	return 0;
}

struct Flow : Module {
	enum ParamIds {
		DIRSCALE_PARAM,
		XYSCALE_PARAM,
		SENS_PARAM,
		VIEWTYPE_PARAM,
		NUM_PARAMS,
	};
	enum InputIds {
		NUM_INPUTS
	};
	enum OutputIds {
		ANGLE_OUTPUT,
		UP_OUTPUT,
		DOWN_OUTPUT,
		LEFT_OUTPUT,
		RIGHT_OUTPUT,
		X_OUTPUT,
		Y_OUTPUT,
		NUM_OUTPUTS
	};
	enum LightIds {
		BLINK_LIGHT,
		NUM_LIGHTS
	};

	Directions directions;
	vector<uchar> image;
	std::atomic<bool> renderDataFree;
	std::atomic<bool> dirty;
	std::atomic<bool> read;
	std::atomic<bool> useCPUorOpenCL;
	std::atomic<float> sens;
	std::atomic<int> viewType;

	RenderData renderData;
	thread opencvThread;

	Flow() : Module(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS) {
		image = vector<uchar>(W*H*4*sizeof(uchar));
		renderDataFree.store(true);
		dirty.store(false);
		read.store(false);
		useCPUorOpenCL.store(false); // USE CPU
		sens.store(0.4);
		viewType.store(0);
		renderData.directions = &directions;
		renderData.image = &image;
		renderData.free = &renderDataFree;
		renderData.dirty = &dirty;
		renderData.read = &read;
		renderData.useCPUorOpenCL = &useCPUorOpenCL;
		renderData.sens = &sens;
		renderData.viewType = &viewType;

		// renderData.width = W;
		// renderData.height = H;

		opencvThread = thread(cameraOpenCVWorker, std::ref(renderData));
		opencvThread.detach();
	}

	void step() {
		// while(!renderDataFree){
		// }

		if(read) {
			read.store(true);
			// cout << directions.up << " " << directions.down << " " << directions.left << " " << directions.right << endl;

			float xyscale = params[XYSCALE_PARAM].value;
			float dirscale = params[DIRSCALE_PARAM].value;
			float sens_ = params[SENS_PARAM].value;
			sens.store(sens_);

			float x = directions.right - directions.left;
		    float y = directions.up - directions.down;
		    // float anglee = fastAtan2(y/10, x/10);

			outputs[ANGLE_OUTPUT].value = (directions.angle/360.0f)*10*xyscale;
			outputs[UP_OUTPUT].value = directions.up * dirscale;
			outputs[DOWN_OUTPUT].value = directions.down * dirscale;
			outputs[LEFT_OUTPUT].value = directions.left * dirscale;
			outputs[RIGHT_OUTPUT].value = directions.right * dirscale;
			outputs[X_OUTPUT].value = x * xyscale;
			outputs[Y_OUTPUT].value = y * xyscale;

			read.store(false);
		}
	}

	void onReset() override {
		useCPUorOpenCL = false;
	}
};

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

		Mat image_alpha = cv::imread(assetPlugin(plugin, "res/preload.png"), IMREAD_UNCHANGED);//IMREAD_COLOR);//
		cout << "Preload Image format " << typeToString(image_alpha.type())  << "isContinuous()" << image_alpha.isContinuous() << endl;

		// cv::cvtColor(image, image, COLOR_BGR2RGBA, 4);
		// int arrayLength = image.total()*4;
		int arrayLength = image_alpha.total()*4;

		vect = vector<uchar>(arrayLength);
		// raiiArray = std::shared_ptr<char>(new char[arrayLength], std::default_delete<char[]>());
		// preloadImage = new uchar[arrayLength];
        // Mat continuousRGBA(image.size(), CV_8UC4, vect.data());
        Mat continuousRGBA(image_alpha.size(), CV_8UC4, vect.data());
		cv::cvtColor(image_alpha, continuousRGBA, COLOR_BGR2RGBA, 4);
		// cv::cvtColor(image, continuousRGBA, COLOR_BGR2RGBA, 4);
	}

	void onMouseDown(EventMouseDown &e) override {
		Widget::onMouseDown(e);
		if (!e.target)
			e.target = this;
		if(module->viewType)
			module->viewType.store(0);
		else
			module->viewType.store(1);
	}

	void draw(NVGcontext *vg) override {
		if(img == 0) {
			img = nvgCreateImageRGBA(vg, 160, 120, 0, vect.data());
		}

		if(module->renderDataFree && module->dirty) {
			module->dirty.store(false);
			nvgUpdateImage(vg, img, module->image.data());
		}

		nvgBeginPath(vg);
		// nvgScale(vg, width/W, height/H);
		NVGpaint imgPaint = nvgImagePattern(vg, 0, 0, 160,120, 0, img, 1.0f);
		nvgRect(vg, 0, 0, 160, 120);
		nvgFillPaint(vg, imgPaint);
		nvgFill(vg);
		nvgClosePath(vg);
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

		addParam(ParamWidget::create<Knob31>(rack::Vec(122, 162), module, Flow::DIRSCALE_PARAM, -1.0, 1.0, 1.0));
		addParam(ParamWidget::create<Knob31>(rack::Vec(122, 299), module, Flow::XYSCALE_PARAM, -1.0, 1.0, 1.0));
		addParam(ParamWidget::create<Knob31>(rack::Vec(14, 162), module, Flow::SENS_PARAM, 0.01, 1, 1.99));

		addOutput(Port::create<PJ301MPort>(rack::Vec(94.5, 272), Port::OUTPUT, module, Flow::ANGLE_OUTPUT));
		addOutput(Port::create<PJ301MPort>(rack::Vec(78, 178), Port::OUTPUT, module, Flow::UP_OUTPUT));
		addOutput(Port::create<PJ301MPort>(rack::Vec(78, 226), Port::OUTPUT, module, Flow::DOWN_OUTPUT));
		addOutput(Port::create<PJ301MPort>(rack::Vec(54, 202), Port::OUTPUT, module, Flow::LEFT_OUTPUT));
		addOutput(Port::create<PJ301MPort>(rack::Vec(102, 202), Port::OUTPUT, module, Flow::RIGHT_OUTPUT));
		addOutput(Port::create<PJ301MPort>(rack::Vec(88.5, 302), Port::OUTPUT, module, Flow::X_OUTPUT));
		addOutput(Port::create<PJ301MPort>(rack::Vec(64.4, 278), Port::OUTPUT, module, Flow::Y_OUTPUT));
	}

	void appendContextMenu(Menu *menu) override {
		Flow *module = dynamic_cast<Flow*>(this->module);

		struct CPUorOpenCLItem : MenuItem {
			Flow *module;
			bool useCPUorOpenCL;
			void onAction(EventAction &e) override {
				module->useCPUorOpenCL.store(useCPUorOpenCL);
			}
		};

		struct ClockItem : MenuItem {
			Flow *module;
			Menu *createChildMenu() override {
				Menu *menu = new Menu();
				std::vector<bool> options = {false, true};
				std::vector<std::string> optionNames = {"Use CPU", "Use GPU (OpenCL)"};
				for (size_t i = 0; i < options.size(); i++) {
					bool useCPUorOpenCL = module->useCPUorOpenCL;
					CPUorOpenCLItem *item =
						MenuItem::create<CPUorOpenCLItem>(optionNames[i],
															CHECKMARK(useCPUorOpenCL == options[i]));
					item->module = module;
					item->useCPUorOpenCL = options[i];
					menu->addChild(item);
				}
				return menu;
			}
		};

		menu->addChild(construct<MenuLabel>());
		ClockItem *item = MenuItem::create<ClockItem>("GPU / CPU");
		item->module = module;
		menu->addChild(item);
	}
};

}

Model *modelFlowModule = Model::create<dsw_vision::Flow, dsw_vision::FlowWidget>("DSW-Vision", "Flow", "Flow", OSCILLATOR_TAG);
