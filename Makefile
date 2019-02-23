# If RACK_DIR is not defined when calling the Makefile, default to two directories above
RACK_DIR ?= ../..

# Must follow the format in the Naming section of
# https://vcvrack.com/manual/PluginDevelopmentTutorial.html
SLUG = DSW

# Must follow the format in the Versioning section of
# https://vcvrack.com/manual/PluginDevelopmentTutorial.html
VERSION = 0.6.0

# FLAGS will be passed to both the C and C++ compiler
FLAGS += -I/usr/local/Cellar/opencv/4.0.1/include/opencv4
# FLAGS += -I/usr/local/include/opencv4

# FLAGS += -I/usr/local/Cellar/opencv@3/3.4.5/include

CFLAGS +=
CXXFLAGS +=

# dynamic linking
# LIBS_opencv = -L/usr/local/Cellar/opencv/4.0.1/lib -lopencv_video
# LIBS_opencv = -L/usr/local/Cellar/opencv/4.0.1/lib -lopencv_imgproc -lopencv_core -lopencv_video tbb.a gfortran.a gomp.a quadmath.a openblasp.a

LIBS_opencv = -Ldep/lib dep/lib/libopencv_objdetect.a dep/lib/libopencv_stitching.a dep/lib/libopencv_video.a dep/lib/libopencv_calib3d.a dep/lib/libopencv_dnn.a dep/lib/libopencv_features2d.a dep/lib/libopencv_gapi.a dep/lib/libopencv_highgui.a dep/lib/libopencv_imgcodecs.a dep/lib/libopencv_imgproc.a dep/lib/libopencv_ml.a dep/lib/libopencv_photo.a dep/lib/libopencv_videoio.a dep/lib/libopencv_core.a dep/lib/libopencv_flann.a \
	-framework Cocoa -framework Accelerate -framework AVFoundation -framework CoreGraphics -framework CoreMedia -framework CoreVideo -framework QuartzCore -framework OpenCL \
	/usr/local/lib/libtbb.a \
	/usr/local/lib/libtbbmalloc.a \
	/usr/local/lib/libjpeg.a \
	/usr/local/lib/libpng.a \
	/usr/local/lib/libtiff.a \
	-Ldep/lib/opencv4/3rdparty \
	dep/lib/opencv4/3rdparty/libippicv.a \
	dep/lib/opencv4/3rdparty/liblibwebp.a \
	dep/lib/opencv4/3rdparty/libquirc.a \
	dep/lib/opencv4/3rdparty/liblibprotobuf.a \
	dep/lib/opencv4/3rdparty/libittnotify.a \
	dep/lib/opencv4/3rdparty/libippiw.a \
	dep/lib/opencv4/3rdparty/libade.a









# LIBS_opencv = -L/usr/local/Cellar/opencv@3/3.4.5/lib/ -lopencv_core -lopencv_video

# LIBS_opencv = -L/usr/local/Cellar/opencv@3/3.4.5/lib/ -lopencv_core.a -lopencv_video.a -lopencv_imgproc.a


# "/Library/Developer/CommandLineTools/usr/bin/ld" -demangle -lto_library /Library/Developer/CommandLineTools/usr/lib/libLTO.dylib -dynamic -dylib -arch x86_64 -macosx_version_min 10.7.0 -undefined dynamic_lookup -undefined dynamic_lookup -o plugin.dylib build/src/AmbientLight.cpp.o build/src/AmbientLightController.cpp.o build/src/Flow.cpp.o build/src/Vision.cpp.o -framework IOKit -framework CoreFoundation -llibopencv_video.a -llibopencv_imgproc.a -llibopencv_core.a -lc++ -lSystem /Library/Developer/CommandLineTools/usr/lib/clang/10.0.0/lib/darwin/libclang_rt.osx.a

# Careful about linking to shared libraries, since you can't assume much about the user's environment and library search path.
# Static libraries are fine.
LDFLAGS += -framework IOKit -framework CoreFoundation $(LIBS_opencv)

# Add .cpp and .c files to the build
SOURCES += $(wildcard src/*.cpp)

# Add files to the ZIP package when running `make dist`
# The compiled plugin is automatically added.
DISTRIBUTABLES += $(wildcard LICENSE*) res

# Include the VCV Rack plugin Makefile framework
include $(RACK_DIR)/plugin.mk
