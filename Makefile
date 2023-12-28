# Compiler settings
CXX := g++
CXXFLAGS := -std=c++11 -O3 -fopenmp -march=native -Iinclude
LDFLAGS := -lboost_timer -lboost_program_options

# Source files and executable names
SRC_DIR := src
MAIN_DIR := main
BIN_DIR := bin

# Executable names
EXECS := kgraph kgraph_RN kgraph_RS kgraph_BJ kgraph_SQ16 kgraph_PLUS

# Define source file groups
KGRAPH_SRC := $(SRC_DIR)/kgraph.cpp $(SRC_DIR)/MemoryUtils.cpp 
KGRAPH_RN_SRC := $(SRC_DIR)/kgraph_RN.cpp $(SRC_DIR)/MemoryUtils.cpp 
KGRAPH_RS_SRC := $(SRC_DIR)/kgraph_RS.cpp $(SRC_DIR)/MemoryUtils.cpp 
KGRAPH_BJ_SRC := $(SRC_DIR)/kgraph_BJ.cpp $(SRC_DIR)/MemoryUtils.cpp 
KGRAPH_SQ16_SRC := $(SRC_DIR)/kgraph_SQ16.cpp $(SRC_DIR)/MemoryUtils.cpp 
KGRAPH_PLUS_SRC := $(SRC_DIR)/kgraph_PLUS.cpp $(SRC_DIR)/MemoryUtils.cpp 

# Targets
all: $(EXECS)

kgraph:
	$(CXX) $(MAIN_DIR)/main.cpp $(KGRAPH_SRC) -o $(BIN_DIR)/$@ $(CXXFLAGS) $(LDFLAGS)

kgraph_RN:
	$(CXX) $(MAIN_DIR)/main.cpp $(KGRAPH_RN_SRC) -o $(BIN_DIR)/$@ $(CXXFLAGS) $(LDFLAGS)

kgraph_RS:
	$(CXX) $(MAIN_DIR)/main.cpp $(KGRAPH_RS_SRC) -o $(BIN_DIR)/$@ $(CXXFLAGS) $(LDFLAGS)

kgraph_BJ:
	$(CXX) $(MAIN_DIR)/main.cpp $(KGRAPH_BJ_SRC) -o $(BIN_DIR)/$@ $(CXXFLAGS) $(LDFLAGS)

kgraph_SQ16:
	$(CXX) -DSHORT $(MAIN_DIR)/main_SQ16.cpp $(KGRAPH_SQ16_SRC) -o $(BIN_DIR)/$@ $(CXXFLAGS) $(LDFLAGS)

kgraph_PLUS:
	$(CXX) -DSHORT $(MAIN_DIR)/main_SQ16.cpp $(KGRAPH_PLUS_SRC) -o $(BIN_DIR)/$@ $(CXXFLAGS) $(LDFLAGS)

# Create bin directory if it does not exist
$(BIN_DIR):
	mkdir $(BIN_DIR)

# Dependencies
$(EXECS): | $(BIN_DIR)

# Clean
clean:
	rm -rf $(BIN_DIR)/* $(EXECS)

.PHONY: all clean