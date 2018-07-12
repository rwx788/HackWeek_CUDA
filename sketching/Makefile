CXX		     := nvcc
CXX_FLAGS    := -std=c++11 -O3

BIN		     := bin
SRC		     := src
INC	         := include /usr/local/include /usr/local/cuda/include /usr/local/include/opencv
INC_PARAMS   := $(addprefix -I,$(INC))
OPEN_CV_LIBS := opencv_core opencv_imgproc opencv_highgui opencv_cudafilters \
               opencv_cudaarithm opencv_gpu opencv_cudaimgproc opencv_videoio \
			   opencv_imgcodecs
LIB		     := lib

LIBRARIES	 := $(addprefix -l,$(OPEN_CV_LIBS))
EXECUTABLE	 := binary

all: $(BIN)/$(EXECUTABLE)

run: clean all
	./$(BIN)/$(EXECUTABLE)

$(BIN)/$(EXECUTABLE): $(SRC)/*.cu
	$(CXX) $(CXX_FLAGS) $(INC_PARAMS) -L$(LIB) $^ -o $@ $(LIBRARIES)

clean:
	-rm $(BIN)/*
