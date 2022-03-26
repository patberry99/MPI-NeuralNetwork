all: Project4.x 

Project4.x: Project4.cpp
	g++ -std=c++11 -mavx -o Project4.x Project4.cpp
