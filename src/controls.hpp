#include "window.hpp"

namespace dsw_vision {

#include "rack.hpp"

using namespace rack;

struct DSWKnob : RoundKnob {
	DSWKnob(const char* svg, int dim) {
		setSVG(SVG::load(assetPlugin(plugin, svg)));
		box.size = Vec(dim, dim);
		shadow->blurRadius = 2.0;
		shadow->box.pos = Vec(0.0, 3.0);
	}
};

struct Knob31 : DSWKnob {
	Knob31() : DSWKnob("res/knob_31px.svg", 29) { }
};

struct SliderSwitch : SVGSwitch, ToggleSwitch {
	SliderSwitch() { }
};

struct SliderSwitch2State35x9 : SliderSwitch {
	SliderSwitch2State35x9() {
		addFrame(SVG::load(assetPlugin(plugin, "res/slider_switch_2_35x9_0.svg")));
		addFrame(SVG::load(assetPlugin(plugin, "res/slider_switch_2_35x9_1.svg")));
		sw->wrap();
		box.size = sw->box.size;
	}
};

struct Port24 : SVGPort {
	Port24() {
		setSVG(SVG::load(assetPlugin(plugin, "res/port.svg")));
		box.size = Vec(24, 24);
		shadow->blurRadius = 1.0;
		shadow->box.pos = Vec(0.0, 1.5);
	}
};

struct SVGPanelWithoutOutline : SVGPanel {
	void step() override {
		if (isNear(rack::gPixelRatio, 1.0)) {
			// Small details draw poorly at low DPI, so oversample when drawing to the framebuffer
			oversample = 2.0;
		}
		FramebufferWidget::step();
	}

	void setBackgroundWithoutOutline(std::shared_ptr<SVG> svg) {
		SVGWidget *sw = new SVGWidget();
		sw->setSVG(svg);
		addChild(sw);

		// Set size
		box.size = sw->box.size.div(RACK_GRID_SIZE).round().mult(RACK_GRID_SIZE);

		// do not draw module outline
	}
};

}