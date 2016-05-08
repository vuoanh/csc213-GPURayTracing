#if !defined(GEOM_HH)
#define GEOM_HH

#include "bitmap.hh"
#include "vec.hh"

//http://stackoverflow.com/questions/6978643/cuda-and-classes
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif



/**
 * An abstract class that represents a shape in the scene
 */
class shape {
public:
  /**
   * Create a new shape
   * \param pos The position of the shape
   */
CUDA_CALLABLE_MEMBER shape(vec pos) : _pos(pos), 
                   _reflectivity(0.95),
                   _diffusion(0.75), 
                   _spec_intensity(0.9), 
                   _spec_density(200) {}
  
  // Getter and setters for the shape's position
CUDA_CALLABLE_MEMBER  vec get_pos() { return _pos; }
CUDA_CALLABLE_MEMBER  void set_pos(vec p) { _pos = p; }
CUDA_CALLABLE_MEMBER  void set_pos(float x, float y, float z) { _pos = vec(x, y, z); }
  
  // Get the color of this shape at a specific point
 CUDA_CALLABLE_MEMBER vec get_color(vec pos) {
    return _color;
  }
  
  // Set this shape to a single static color
 CUDA_CALLABLE_MEMBER void set_color(vec c) {
    _color = c;
  }
  
  // Getter and setter for reflectivity
 CUDA_CALLABLE_MEMBER float get_reflectivity() { return _reflectivity; }
 CUDA_CALLABLE_MEMBER void set_reflectivity(float r) { _reflectivity = r; }
  
  // Getter and setter for the diffusion level
 CUDA_CALLABLE_MEMBER float get_diffusion() { return _diffusion; }
 CUDA_CALLABLE_MEMBER void set_diffusion(float d) { _diffusion = d; }
  
  // Getter and setter for the specular highlight intensity
 CUDA_CALLABLE_MEMBER float get_spec_intensity() { return _spec_intensity; }
 CUDA_CALLABLE_MEMBER void set_spec_intensity(float s) { _spec_intensity = s; }
  
  // Getter and setter for the specular highlight density
 CUDA_CALLABLE_MEMBER float get_spec_density() { return _spec_density; }
 CUDA_CALLABLE_MEMBER void set_spec_density(float s) { _spec_density = s; }
  
  // Abstract method for the intersection calculation (shape-dependent)
 CUDA_CALLABLE_MEMBER virtual float intersection(vec origin, vec dir) = 0;
  
  // Abstract method to compute a normal at a point (shape-dependent)
 CUDA_CALLABLE_MEMBER virtual vec normal(vec p) = 0;
    
protected:
  vec _pos;
  vec _color;
  float _reflectivity;
  float _diffusion;
  float _spec_intensity;
  float _spec_density;
};

/**
 * A sphere at a given position and radius
 */
class sphere : public shape {
public:
  /**
   * Create a sphere at a position with a given radius
   * \param pos     The center of the sphere
   * \param radius  The radius of the sphere
   */
 CUDA_CALLABLE_MEMBER sphere(vec pos, float radius) : shape(pos), _radius(radius) {}

  CUDA_CALLABLE_MEMBER sphere() : shape(vec()), _radius(0) {}
  
  // Getter and setter for the radius
 CUDA_CALLABLE_MEMBER float get_radius() { return _radius; }
   CUDA_CALLABLE_MEMBER vec get_pos() { return _pos; }
 CUDA_CALLABLE_MEMBER void set_radius(float r) { _radius = r; }
  
  // Intersection calculation
 CUDA_CALLABLE_MEMBER virtual float intersection(vec origin, vec dir) {
    vec v = origin - _pos;
    
    // Next, we solve the quadratic equation that tells us where
    // our viewing angle d intersects with the sphere's surface.
    // First, the piece of the quadratic equation under the sqrt:
    float squared_portion = v.dot(dir) * v.dot(dir) - (v.dot(v) - _radius * _radius);
    
    // If the squared portion is negative, there is no intersection
    if(squared_portion >= 0) {
      // Compute both the plus and minus terms in the quadratic equation
      float t_plus = -v.dot(dir) + sqrt(squared_portion);
      float t_minus = -v.dot(dir) - sqrt(squared_portion);
      
      if(t_plus >= 0 && t_plus < t_minus) {
        return t_plus;
      } else if(t_minus >= 0) {
        return t_minus;
      }
    }
    
    return -1;
  }
  
  // Normal calculation
  CUDA_CALLABLE_MEMBER virtual vec normal(vec p) {
    return (p - _pos).normalized();
  }
  
private:
  float _radius;
};

/**
 * An infinite plane described by a normal vector and origin
 */
class plane : public shape {
public:
  /**
   * Create a plane
   * \param pos   The origin point of this plane
   * \param norm  A vector that points perpendicular to the face of this plane
   */
 CUDA_CALLABLE_MEMBER plane(vec pos, vec norm) : shape(pos), _norm(norm) {}
  CUDA_CALLABLE_MEMBER plane() : shape(vec()), _norm(vec()) {}
  
  // Getter and setter for the plane normal vector
 CUDA_CALLABLE_MEMBER vec get_norm() { return _norm; }
 CUDA_CALLABLE_MEMBER void set_norm(vec n) { _norm = n; }
  
 CUDA_CALLABLE_MEMBER virtual float intersection(vec origin, vec dir) {
    // Dot the plane normal with the ray direction
    float d_dot_norm = _norm.dot(dir);
    
    // If the dot product is zero, the ray does not intersect
    if(d_dot_norm != 0) {
      float distance = (_pos - origin).dot(_norm) / d_dot_norm;
      if(distance >= 0) return distance;
    }
    
    return -1;
  }
  
  // Get the normal for this plane at a given point (identical everywhere)
  CUDA_CALLABLE_MEMBER virtual vec normal(vec p) {
    return _norm;
  }
  
private:
  vec _norm;
};

/**
 * A viewport class that maps from pixels on a viewing plane to a ray direction
 */
class viewport {
public:
  /**
   * Create a new viewport
   * \param origin    The position of the viewport's "eye"
   * \param look_dir  A vector that points from the "eye" to the middle of the viewing plane
   * \param up        A vector that points up
   * \param width     The width of the viewing screen
   * \param height    The height of the viewing screen
   */
 CUDA_CALLABLE_MEMBER viewport(vec origin, vec look_dir, vec up, float width, float height) :
      _origin(origin), _look_dir(look_dir), _width(width), _height(height) {
    
    // Compute a viewing direction unit vector
    vec unit_dir = _look_dir.normalized();
    
    // Make a unit vector perpendicular to the viewing direction.
    _up = (up - unit_dir * up.dot(unit_dir)).normalized();
    
    // Make a unit vector perpendicular to both up and the viewing direction
    _right = _up.rotated_around(unit_dir, M_PI_2).normalized();
    
    // Scale _up and _right
    _up /= fmin(width, height);
    _right /= fmin(width, height);
  }
  
  // Get the origin of the viewpoert
 CUDA_CALLABLE_MEMBER vec origin() {
    return _origin;
  }
  
  // Get a ray from the origin through a specific viewing plane coordinate
 CUDA_CALLABLE_MEMBER vec dir(float x, float y) {
    // Compute the point on the viewing plane that we are looking through
    vec p = _look_dir - _right * (x - _width / 2) + _up * (_height / 2 - y);
    
    // Return a unit vector pointing from the view origin to this point
    return p.normalized();
  }
  
private:
  vec _origin;
  vec _look_dir;
  vec _up;
  vec _right;
  float _width;
  float _height;
};

#endif
