#ifndef AmbientLightController_MAC
#define AmbientLightController_MAC

#include <memory>

namespace dsw_vision {

#include <IOKit/IOKitLib.h>

class AmbientLightController {
public:
	static std::unique_ptr<AmbientLightController> Create();
	bool ReadSensorValue(uint64_t lux_value[2]);
	~AmbientLightController();
private:
	AmbientLightController();
	bool Init();
	io_connect_t io_connection_;
};

}


#endif