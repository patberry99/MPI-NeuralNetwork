all: Project4

Project4: Project4.cpp
	mpiCC  -std=c++11 -o Project4 Project4.cpp
