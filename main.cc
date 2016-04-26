#include <cmath>
#include <cstdio>
#include <vector>
#include <pthread.h>
#include <future>
#include <SDL.h>

#include "bitmap.hh"
#include "geom.hh"
#include "gui.hh"
#include "util.hh"
#include "vec.hh"

// Screen size
#define WIDTH 640
#define HEIGHT 480

// Rendering Properties
#define AMBIENT 0.3         // Ambient illumination
#define OVERSAMPLE 2        // Sample 2x2 subpixels
#define MAX_REFLECTIONS 10  // The maximum number of times a ray is reflected
#define EPSILON 0.03        // Shift points off surfaces by this much
// Create threads for oversampling
#define num_over_thread OVERSAMPLE*OVERSAMPLE

using namespace std;

// Set up the 3D scene
void init_scene();

// Trace a ray through the scene to determine its color
vec raytrace(vec origin, vec dir, size_t reflections);

// A list of shapes that make up the 3D scene. Initialized by init_scene
std::vector<shape*> scene;

// A list of light positions, all emitting pure white light
std::vector<vec> lights;

// computes the color for the quadrants
void set_quadrant_color(int xlow, int xhigh, viewport view, float yrot, vec result_array[][HEIGHT]);

/**
 * Entry point for the raytracer
 * \param argc  The number of command line arguments
 * \param argv  An array of command line arguments
 */
int main(int argc, char** argv) {
  // Create a GUI window
  gui ui("Raytracer", WIDTH, HEIGHT);
  
  // Initialize the 3D scene
  init_scene();
  
  // Set up the viewport
  viewport view(vec(0, 100, -300), // Look from here
                vec(0, -0.25, 1),  // Look in this direction
                vec(0, 1, 0),      // Up is up
                WIDTH,             // Use screen width
                HEIGHT);           // Use screen height
  
  // Save the starting time
  size_t start_time = time_ms();
  
  bool running = true;

  
  // Loop until we get a quit event
  while(running) {
    // Process events
    SDL_Event event;
    while(SDL_PollEvent(&event) == 1) {
      // If the event is a quit event, then leave the loop
      if(event.type == SDL_QUIT) running = false;
    }
    
    // Rotate the camera around the scene once every five seconds
    float yrot = (time_ms() - start_time)/5000.0 * M_PI * 2;
    
    // Render the frame to this bitmap
    bitmap bmp(WIDTH, HEIGHT);
    vec result_array[WIDTH][HEIGHT];  
    // Loop over all pixels in the bitmap
    thread q1 = thread(set_quadrant_color, 0, WIDTH/4, view, yrot, result_array);
    thread q2 = thread(set_quadrant_color, WIDTH/4, 2*WIDTH/4, view, yrot,result_array );
    thread q3 = thread(set_quadrant_color, 2*WIDTH/4, 3*WIDTH/4, view, yrot,result_array );
    thread q4 = thread(set_quadrant_color, 3*WIDTH/4, WIDTH, view, yrot,result_array );
    
    q1.join();
    q2.join();
    q3.join();
    q4.join();
    
    for (int x = 0 ; x < WIDTH; x++){
      for(int y = 0; y < HEIGHT; y++){
        bmp.set(x, y, result_array[x][y]);
      }
    }

    // Display the rendered frame
    ui.display(bmp);
  }
  
  return 0;
}
// computes the color for the quadrants
void set_quadrant_color(int xlow, int xhigh, viewport view, float yrot, 
    vec result_array[][HEIGHT]){
  for(int y = 0; y < HEIGHT; y++) {
    for(int x = xlow; x < xhigh; x++) {
      // Next, we collect several colors for rays that all correspond to this 
      // pixel. When we average those rays, we get a smoother image.
      
     
      // Colors will be added here, then scaled down by OVERSAMPLE^2
      vec result;
      future<vec> future_result[num_over_thread];
      int thread_count = -1;
      int rc;
      // Loop over y subpixel positions
      for(int y_sample = 0; y_sample < OVERSAMPLE; y_sample++) {
        // The y offset is half way between the edges of this subpixel
        float y_off = (y_sample + 0.5) / OVERSAMPLE;
        
        // Loop over x subpixel positions
        for(int x_sample = 0; x_sample < OVERSAMPLE; x_sample++) {
          // The x offset is half way between the edges of this subpixel
          float x_off = (x_sample + 0.5) / OVERSAMPLE;
          
	  result += raytrace(view.origin().yrotated(yrot),view.dir(x + x_off, y + y_off).yrotated(yrot), 0 );
        }
      }
      
      // Average the oversampled points
      result /= OVERSAMPLE * OVERSAMPLE;
   
      // Set the pixel color
      result_array[x][y] = result;
    }
  }
}
/**
 * Follow a ray backwards through the scene and return the ray's color
 * \param origin        The origin of the ray
 * \param dir           The direction of the ray
 * \param reflections   The number of times this ray has been reflected
 * \returns             The color of this ray
 */
vec raytrace(vec origin, vec dir, size_t reflections) {
  // Normalize the direction vector
  dir = dir.normalized();
  
  // Keep track of the closest shape that is intersected by this ray
  shape* intersected = NULL;
  float intersect_distance = 0;
  
  // Loop over all shapes in the scene to find the closest intersection
  for(shape* shape : scene) {
    float distance = shape->intersection(origin, dir);
    if(distance >= 0 && (distance < intersect_distance || intersected == NULL)) {
      intersect_distance = distance;
      intersected = shape;
    }
  }
  
  // If the ray didn't intersect anything, just return the ambient color
  if(intersected == NULL) return vec(AMBIENT, AMBIENT, AMBIENT);
  
  // The new starting point for the reflected ray is the point of intersection.
  // Find the reflection point just a *little* closer so it isn't on the object.
  // Otherwise, the new ray may intersect the same shape again depending on
  // rounding error.
  vec intersection = origin + dir * (intersect_distance - EPSILON);
  
  // Initialize the result color to the ambient light reflected in the shapes color
  vec result = intersected->get_color(intersection) * AMBIENT;
  
  // Add recursive reflections, unless we're at the recursion bound
  if(reflections < MAX_REFLECTIONS) {
    // Find the normal at the intersection point
    vec n = intersected->normal(intersection);

    // Reflect the vector across the normal
    vec new_dir = dir - n * 2.0 * n.dot(dir);
      
    // Compute the reflected color by recursively raytracing from this point
    vec reflected = raytrace(intersection, new_dir, reflections + 1);
  
    // Add the reflection to the result, tinted by the color of the shape
    result += reflected.hadamard(intersected->get_color(intersection)) *
      intersected->get_reflectivity();
    
    // Add the contribution from all lights in the scene
    for(vec& light : lights) {
      // Create a unit vector from the intersection to the light source
      vec shadow_dir = (light - intersection).normalized();

      // Check to see if the shadow vector intersects the scene
      bool in_shadow = false;
      for(shape* shape : scene) {
        if(shape->intersection(intersection, shadow_dir) >= 0) {
          in_shadow = true;
          break;
        }
      }
    
      // If there is a clear path to the light, add illumination
      if(!in_shadow) {
        // Compute the intensity of the diffuse lighting
        float diffuse_intensity = intersected->get_diffusion() *
          fmax(0, n.dot(shadow_dir));
      
        // Add diffuse lighting tinted by the color of the shape
        result += intersected->get_color(intersection) * diffuse_intensity;
        
        // Find the vector that bisects the eye and light directions
        vec bisector = (shadow_dir - dir).normalized();

        // Compute the intensity of the specular reflections, which are not affected by the color of the object
        float specular_intensity = intersected->get_spec_intensity() *
          fmax(0, pow(n.dot(bisector), (int)intersected->get_spec_density()));
      
        // Add specular highlights
        result += vec(1.0, 1.0, 1.0) * specular_intensity;
      }
    }
  }
  
  return result;
}

/**
 * Add objects and lights to the scene.
 * Creates three spheres, a flat plane, and two light sources
 */
void init_scene() {
  // Add a red sphere
  sphere* red_sphere = new sphere(vec(60, 50, 0), 50);
  red_sphere->set_color(vec(0.75, 0.125, 0.125));
  red_sphere->set_reflectivity(0.5);
  scene.push_back(red_sphere);
  
  // Add a green sphere
  sphere* green_sphere = new sphere(vec(-15, 25, -25), 25);
  green_sphere->set_color(vec(0.125, 0.6, 0.125));
  green_sphere->set_reflectivity(0.5);
  scene.push_back(green_sphere);
  
  // Add a blue sphere
  sphere* blue_sphere = new sphere(vec(-50, 40, 75), 40);
  blue_sphere->set_color(vec(0.125, 0.125, 0.75));
  blue_sphere->set_reflectivity(0.5);
  scene.push_back(blue_sphere);
  
  // Add a flat surface
  plane* surface = new plane(vec(0, 0, 0), vec(0, 1, 0));
  // The following line uses C++'s lambda expressions to create a function
  surface->set_color([](vec pos) {
    // This function produces a grid pattern on the plane
    if((int)pos.x() % 100 == 0 || (int)pos.z() % 100 == 0) {
      return vec(0.3, 0.3, 0.3);
    } else {
      return vec(0.15, 0.15, 0.15);
    }
  }); 
  surface->set_diffusion(0.25);
  surface->set_spec_density(10);
  surface->set_spec_intensity(0.1);
  scene.push_back(surface);
  
  // Add two lights
  lights.push_back(vec(-1000, 300, 0));
  lights.push_back(vec(100, 900, 500));
}
