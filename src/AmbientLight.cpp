#include "Vision.hpp"
#include "AmbientLightController.hpp"

#include "controls.hpp"

namespace dsw_vision {

struct AmbientLight : Module {
	enum ParamIds {
		OFFSET_PARAM,
		SCALE_PARAM,
		UNIBI_PARAM,
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

	std::unique_ptr<AmbientLightController> ambient_light_controller;
	bool ambient_light_sensor_available;
	float checkPhase = 0.0;
	float lux = 1.0;
	float deltaTime = 0.0;
	bool unibi = false;

	AmbientLight() : Module(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS) {
		ambient_light_controller = AmbientLightController::Create();
		ambient_light_sensor_available = ambient_light_controller.get() != nullptr;
		deltaTime = engineGetSampleRate()/240.0 * engineGetSampleTime();
	}
	void step() override;
	double readLight();

	// For more advanced Module features, read Rack's engine.hpp header file
	// - toJson, fromJson: serialization of internal data
	// - onSampleRateChange: event triggered by a change of sample rate
	// - onReset, onRandomize, onCreate, onDelete: implements special behavior when user clicks these from the context menu
};

double LMUvalueToLux(uint64_t raw_value) {
  // Conversion formula from regression.
  // https://bugzilla.mozilla.org/show_bug.cgi?id=793728
  // Let x = raw_value, then
  // lux = -2.978303814*(10^-27)*x^4 + 2.635687683*(10^-19)*x^3 -
  //       3.459747434*(10^-12)*x^2 + 3.905829689*(10^-5)*x - 0.1932594532

  static const long double k4 = pow(10.L, -7);
  static const long double k3 = pow(10.L, -4);
  static const long double k2 = pow(10.L, -2);
  static const long double k1 = pow(10.L, 5);
  long double scaled_value = raw_value / k1;

  long double lux_value =
      (-3 * k4 * pow(scaled_value, 4)) + (2.6 * k3 * pow(scaled_value, 3)) +
      (-3.4 * k2 * pow(scaled_value, 2)) + (3.9 * scaled_value) - 0.19;

  double lux = ceil(static_cast<double>(lux_value));
  return lux > 0 ? lux : 0;
}

double AmbientLight::readLight(){
	if (!ambient_light_sensor_available)
		return 1;
    uint64_t lux_value[2];
    auto controller = ambient_light_controller.get();
	if (!controller->ReadSensorValue(lux_value))
		return 1;
	uint64_t mean = (lux_value[0] + lux_value[1]) / 2;
	double lux = LMUvalueToLux(mean);

	// info("%d, %d; MEAN %d LUX %f", lux_value[0], lux_value[1], mean, lux);
	return lux;
}

void AmbientLight::step() {
	unibi = (params[UNIBI_PARAM].value > 0.0f);
	float offset = params[OFFSET_PARAM].value;
	float scale  = params[SCALE_PARAM].value;

	checkPhase += deltaTime;
	if(checkPhase >= 1.0f){
		checkPhase -= 1.0f;
		lux = readLight();

		float internal_adj = 50;
		float out;

		out = lux / internal_adj;
		// out = pow(out, 1.0/4.0);

		scale = scale < 0.0f ? -pow(scale, 2.0f) : pow(scale, 2.0f);
		scale *= 10.0;

		out += 10.0f * offset;
		out *= scale;

		// if (!_disableOutputLimit) {
		// 	out = clamp(out, -12.0f, 12.0f);
		// }
		lux = out;


		// // Compute the frequency from the pitch parameter and input
		// float sensitivity = params[ADJ_PARAM].value;
		// // float adjusted = lux / sensitivity;
		// float adjusted = rescale(lux, 0, sensitivity, -5.0f, 5.0f);
		// // info("%f -- %f, %f, %f", lux, sensitivity, adjusted);

		outputs[LIGHT_OUTPUT].value = lux;
		// lights[BLINK_LIGHT].value = abs(adjusted);
	}
}

struct VCA_1VUKnob : Knob {
	VCA_1VUKnob() {
		box.size = Vec(16, 220);
	}

	void draw(NVGcontext *vg) override {
		nvgBeginPath(vg);
		nvgRoundedRect(vg, 0, 0, box.size.x, box.size.y, 2.0);
		nvgFillColor(vg, nvgRGB(0, 0, 0));
		nvgFill(vg);

		AmbientLight *module = dynamic_cast<AmbientLight*>(this->module);

		const int segs = 55;
		const Vec margin = Vec(1, 1);
		rack::Rect r = box.zeroPos().shrink(margin);

		for (int i = 0; i < segs; i++) {
			float segValue = clamp(value * segs - (segs - i - 1), 0.f, 1.f);
			float amplitude = value * module->lux;
			float segAmplitude = clamp(amplitude * segs - (segs - i - 1), 0.f, 1.f);
			nvgBeginPath(vg);
			nvgRect(vg, r.pos.x, r.pos.y + r.size.y / segs * i + 0.5,
				r.size.x, r.size.y / segs - 1.0);
			// if (segValue > 0.f) {
			// 	nvgFillColor(vg, colorAlpha(nvgRGBf(0.33, 0.33, 0.33), segValue));
			// 	nvgFill(vg);
			// }
			// if (segAmplitude > 0.f) {
			{
				nvgFillColor(vg, colorAlpha(COLOR_BLUE, segAmplitude));
				nvgFill(vg);
			}
		}
	}
};

struct AmbientLightWidget : ModuleWidget {
	AmbientLightWidget(AmbientLight *module) : ModuleWidget(module) {
		setPanel(SVG::load(assetPlugin(plugin, "res/ALV.svg")));

		addParam(ParamWidget::create<Knob31>(Vec(7.9, 197.8), module, AmbientLight::OFFSET_PARAM, -1.0, 1.0, 0.0));
		addParam(ParamWidget::create<Knob31>(Vec(7.9, 262.8), module, AmbientLight::SCALE_PARAM, -1.0, 1.0, 0.2));

		addOutput(Port::create<Port24>(Vec(10.376, 320.801), Port::OUTPUT, module, AmbientLight::LIGHT_OUTPUT));
	}


	// void drawShadow(NVGcontext *vg) override {
	// }

	// void draw(NVGcontext *vg) override { Widget::draw(vg); }
};

}

// Specify the Module and ModuleWidget subclass, human-readable
// author name for categorization per plugin, module slug (should never
// change), human-readable module name, and any number of tags
// (found in `include/tags.hpp`) separated by commas.
Model *modelLightModule = Model::create<dsw_vision::AmbientLight, dsw_vision::AmbientLightWidget>("DSW-Vision", "ALV", "ALV Ambient Light Voltage", OSCILLATOR_TAG);
