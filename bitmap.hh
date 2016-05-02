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
class bitmap {
public:
  // Constructor: set up the bitmap width, height, and data array
  bitmap(size_t width, size_t height) : _width(width), _height(height) {
    _data = new rgb32[width*height];
  }
  
  // Destructor: free the data array
  ~bitmap() {
   delete _data;
  }
  
  // Get the size of this bitmap's image data
  size_t size() { return _width*_height*sizeof(rgb32); }
  
  // Copy this bitmap to a given data location
  void copy_to(void* dest) {
    memcpy(dest, _data, size());
  }
  
  // Get the height of the bitmap
  CUDA_CALLABLE_MEMBER size_t height() { return _height; }
  
  // Get the width of the bitmap
  CUDA_CALLABLE_MEMBER size_t width() { return _width; }
  
  // Set the color at a given location
  CUDA_CALLABLE_MEMBER void set(size_t x, size_t y, vec c) {
    assert(x < _width && "X coordinate is out of bounds");
    assert(y < _height && "Y coordinate is out of bounds");
    _data[y*_width+x].alpha = 0;
    _data[y*_width+x].blue = (uint8_t)fmax(0, fmin(255, 255*c.x()));
    _data[y*_width+x].green = (uint8_t)fmax(0, fmin(255, 255*c.y()));
    _data[y*_width+x].red = (uint8_t)fmax(0, fmin(255, 255*c.z()));
  }
  
private:
  size_t _width;
  size_t _height;
  
  struct rgb32 {
    uint8_t alpha;
    uint8_t blue;
    uint8_t green;
    uint8_t red;
  };
  
  rgb32* _data;
};

#endif
