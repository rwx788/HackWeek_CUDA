CXX		  := g++
CXX_FLAGS :=

BIN		   := bin
SRC		   := src
INC	       := include /usr/local/cuda/include
INC_PARAMS := $(addprefix -I,$(INC))
LIB		   := lib

LIBRARIES	:=
EXECUTABLE	:= main

all: $(BIN)/$(EXECUTABLE)

run: clean all
	./$(BIN)/$(EXECUTABLE)

$(BIN)/$(EXECUTABLE): $(SRC)/*.cpp
	$(CXX) $(CXX_FLAGS) $(INC_PARAMS) -L$(LIB) $^ -o $@ $(LIBRARIES)

clean:
	-rm $(BIN)/*
