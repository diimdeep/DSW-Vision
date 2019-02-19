#include "Vision.hpp"

#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"

#include <iostream>
#include <iomanip>
#include <vector>

namespace dsw_vision {

using namespace std;
using namespace cv;
using std::cout; using std::cerr; using std::endl;

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
	VideoCapture capture; // open the first camera
	Ptr<DenseOpticalFlow> alg;
	UMat prevFrame, frame, input_frame, flow;

	Flow() : Module(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS) {
		std::cout << "Opening camera..." << std::endl;
		capture = VideoCapture(0);
	    if (!capture.isOpened())
	    {
	        std::cerr << "ERROR: Can't initialize camera capture" << std::endl;
	    }
		alg = FarnebackOpticalFlow::create();
		ocl::setUseOpenCL(false);

		capture >> frame;

        cvtColor(frame, frame, COLOR_BGR2GRAY);
        imshow("frame", frame);
	}
	void step() override;
};

void Flow::step() {
	// unibi = (params[UNIBI_PARAM].value > 0.0f);
	outputs[LIGHT_OUTPUT].value = 1.0;

	// if (!cap.read(input_frame) || input_frame.empty())
 //    {
 //        cout << "Finished reading: empty frame" << endl;
 //        break;
 //    }
}

struct FlowWidget : ModuleWidget {
	FlowWidget(Flow *module) : ModuleWidget(module) {
		setPanel(SVG::load(assetPlugin(plugin, "res/Flow.svg")));
	}
};

}

Model *modelFlowModule = Model::create<dsw_vision::Flow, dsw_vision::FlowWidget>("DSW", "Vision", "Flow", OSCILLATOR_TAG);
