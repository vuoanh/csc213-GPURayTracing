#if !defined(BITMAP_HH)
#define BITMAP_HH

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "vec.hh"
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

struct rgb32 {
  uint8_t alpha;
  uint8_t blue;
  uint8_t green;
  uint8_t red;
} __attribute__((packed));

#define WIDTH 640
#define HEIGHT 480


class bitmap {
public:
  // Constructor: set up the bitmap width, height, and data array
  bitmap() {}
  
  // Get the size of this bitmap's image data
  size_t size() { return WIDTH * HEIGHT * sizeof(rgb32); }
  
  // Copy this bitmap to a given data location
  void copy_to(void* dest) {
    memcpy(dest, _data, size());
  }
  
  // Get the height of the bitmap
  CUDA_CALLABLE_MEMBER size_t height() { return HEIGHT; }
  
  // Get the width of the bitmap
  CUDA_CALLABLE_MEMBER size_t width() { return WIDTH; }
  
  // Set the color at a given location
  CUDA_CALLABLE_MEMBER void set(size_t x, size_t y, vec c) {
    _data[y*WIDTH+x].alpha = 0;
    _data[y*WIDTH+x].blue = (uint8_t)fmax(0, fmin(255, 255*c.z()));
    _data[y*WIDTH+x].green = (uint8_t)fmax(0, fmin(255, 255*c.y()));
    _data[y*WIDTH+x].red = (uint8_t)fmax(0, fmin(255, 255*c.x()));
  }

  CUDA_CALLABLE_MEMBER rgb32 get(size_t x, size_t y) {
    return _data[y*WIDTH+x];
  }
  
private:
  rgb32 _data[WIDTH*HEIGHT];
};

#endif
