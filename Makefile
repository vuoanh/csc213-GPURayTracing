NVCC := nvcc -arch sm_20
NVCC_FLAGS := -g -I/home/curtsinger/include/SDL2 -D_REENTRANT -L/home/curtsinger/lib -lSDL2

all: raytracer

clean:
	@rm -f raytracer

raytracer : main.cu
	$(NVCC) $(NVCC_FLAGS) -o raytracer main.cu

#run: raytracer
#	LD_LIBRARY_PATH=/home/curtsinger/lib cat /home/curtsinger/data/tweets.json | ./twitter
#ROOT:= .
#include $(ROOT)/common.mk
