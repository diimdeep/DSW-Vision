RACK_DIR ?= ../../..
.NOTPARALLEL:

include $(RACK_DIR)/arch.mk

ifdef ARCH_MAC
	opencv = lib/opencv_imgproc.a lib/opencv_core.a lib/opencv_video.a
endif

DEPS += $(opencv)

WGET := wget -c -nc
UNTAR := tar xf

# The install location for `make install`
DEP_LOCAL ?= .
DEP_PATH := $(shell pwd)/$(DEP_LOCAL)

CONFIGURE := ./configure --prefix="$(DEP_PATH)"
ifdef ARCH_WIN
	CMAKE := cmake -G 'MSYS Makefiles' -DCMAKE_INSTALL_PREFIX="$(DEP_PATH)"
else
	CMAKE := cmake -DCMAKE_INSTALL_PREFIX="$(DEP_PATH)"
endif

$(opencv):
	$(WGET) "https://github.com/opencv/opencv/archive/4.0.1.tar.gz"
	$(UNTAR) "4.0.1.tar.gz"
	cd "opencv-4.0.1" && mkdir -p build
	cd "opencv-4.0.1"/build && $(CMAKE) .. \
		-DBUILD_SHARED_LIBS=OFF \
		-DWITH_CSTRIPES=ON \
		-DWITH_OPENCL=ON \
		-DCMAKE_BUILD_TYPE=RELEASE \
		-DBUILD_DOCS=OFF \
		-DBUILD_EXAMPLES=OFF \
		-DCMAKE_OSX_DEPLOYMENT_TARGET= \
		-DBUILD_JASPER=OFF \
		-DBUILD_JPEG=OFF \
		-DBUILD_OPENEXR=OFF \
		-DBUILD_PERF_TESTS=OFF \
		-DBUILD_PNG=OFF \
		-DBUILD_TESTS=OFF \
		-DBUILD_TIFF=OFF \
		-DBUILD_ZLIB=OFF \
		-DBUILD_opencv_hdf=OFF \
		-DBUILD_opencv_java=OFF \
		-DBUILD_opencv_text=OFF \
		-DOPENCV_ENABLE_NONFREE=ON \
		-DOPENCV_GENERATE_PKGCONFIG=OFF \
		-DWITH_1394=OFF \
		-DWITH_CUDA=OFF \
		-DWITH_EIGEN=OFF \
		-DWITH_FFMPEG=OFF \
		-DWITH_GPHOTO2=OFF \
		-DWITH_GSTREAMER=OFF \
		-DWITH_JASPER=OFF \
		-DWITH_OPENEXR=OFF \
		-DWITH_OPENGL=OFF \
		-DWITH_QT=OFF \
		-DWITH_TBB=ON \
		-DWITH_VTK=OFF \
		-DBUILD_opencv_python2=OFF \
		-DBUILD_opencv_python3=OFF \
		-DCPU_DISPATCH=SSE4_1,SSE4_2,AVX \
		-DENABLE_AVX=OFF \
		-DENABLE_AVX2=OFF
		# -DCMAKE_INSTALL_PREFIX="$(realpath $(DEP_LOCAL))/lib"
	$(MAKE) -C "opencv-4.0.1"/build
	$(MAKE) -C "opencv-4.0.1"/build install

dep: $(DEPS)

.PHONY: dep
