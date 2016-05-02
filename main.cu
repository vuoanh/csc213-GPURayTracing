// Remember to ask Charlie:
// 1. Can a CUDA helper function return anything?
// 2. MemCpy inside the kernel?
// 3. Removed the "delete;" lines from the classes (bitmap), still getting the C++ error;
// Need to hardcore num scene objects
// Change that vector to an array
// No For Each

#include <math.h>
#include <stdio.h>
#include <vector>
#include <pthread.h>
#include <SDL.h>

#include "bitmap.hh"
#include "geom.hh"
#include "gui.hh"
#include "util.hh"
#include "vec.hh"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif


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
#define THREADS_PER_BLOCK 8;
#define OBJ_NUM 4
#define LIGHT_NUM 2
using namespace std;

// Set up the 3D scene
void init_scene();

// Trace a ray through the scene to determine its color
CUDA_CALLABLE_MEMBER vec raytrace(vec origin, vec dir, size_t reflections, shape** gpu_scene);

// A list of shapes that make up the 3D scene. Initialized by init_scene
shape* scene[OBJ_NUM];

// A list of light positions, all emitting pure white light
vec lights[LIGHT_NUM];

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

  // GPU shapes
  shape** gpu_scene;
  if (cudaMalloc(&gpu_scene, sizeof(shape*) * OBJ_NUM) != cudaSuccess) {
    fprintf( stderr, "Fail to allocate GPU objects\n");
  }
  if(cudaMemcpy(gpu_scene, scene, sizeof(shape*) * OBJ_NUM, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf( stderr, "Fail to copy objects to GPU\n");
  }

    
  // GPU lights
 vec* gpu_lights;
if (cudaMalloc(&gpu_lights, sizeof(vec) * LIGHT_NUM)!= cudaSuccess) {
    fprintf( stderr, "Fail to allocate GPU lights\n");
  }
  if(cudaMemcpy(gpu_lights, lights,sizeof(vec) * LIGHT_NUM, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf( stderr, "Fail to copy lights to GPU\n");
  }
    
  
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
    bitmap cpu_bmp(WIDTH, HEIGHT);
    bitmap* gpu_bmp;
    vec cpu_result_array[WIDTH][HEIGHT];
    vec* gpu_result_array;
    
    // Allocate memory for the gpu bitmap and the gpu result array
    if (cudaMalloc(&gpu_bmp, sizeof(bitmap))!= cudaSuccess) {
      fprintf( stderr, "Fail to allocate GPU bitmap\n");
    }
    if (cudaMalloc(&gpu_result_array, sizeof(vec) * WIDTH * HEIGHT)!= cudaSuccess) {
      fprintf( stderr, "Fail to allocate GPU result_array\n");
    }

    // Copy memory from the cpu bitmap and result array to the gpu counterparts
    if(cudaMemcpy(gpu_bmp, cpu_bmp, sizeof(bitmap), cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf( stderr, "Fail to copy bitmap to GPU\n");
    }
    // why are we copying from cpu_result array to gpu?
    if(cudaMemcpy(gpu_result_array, cpu_result_array, sizeof(vec) * WIDTH * HEIGHT, cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf( stderr, "Fail to copy result_array to GPU\n");
    }

    // allocating necessary variables for raytrace

    // viewport
    viewport* gpu_viewport;
    if (cudaMalloc(&gpu_viewport, sizeof(viewport))!= cudaSuccess) {
      fprintf( stderr, "Fail to allocate GPU viewport\n");
    }
    if(cudaMemcpy(gpu_viewport, view, sizeof(viewport), cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf( stderr, "Fail to copy viewport to GPU\n");
    }

    /** Potentially useless, probably
     // vec origin    
     vec cpu_origin = view.origin();
     vec* gpu_origin;
     if (cudaMalloc(&gpu_origin, sizeof(vec))!= cudaSuccess) {
     fprintf( stderr, "Fail to allocate GPU vec_origin\n");
     }
     if(cudaMemcpy(gpu_origin, cpu_origin, sizeof(vec), cudaMemcpyHostToDevice) != cudaSuccess) {
     fprintf( stderr, "Fail to copy vec_origin to GPU\n");
     }

     // vec dir
     vec* gpu_dir;
     if (cudaMalloc(&gpu_dir, sizeof(vec))!= cudaSuccess) {
     fprintf( stderr, "Fail to allocate GPU vec_dir\n");
     }
    
     // size_t reflections
     size_t cpu_reflections = 0;
     size_t* gpu_reflections;
     if (cudaMalloc(&gpu_reflections, sizeof(size_t))!= cudaSuccess) {
     fprintf( stderr, "Fail to allocate GPU gpu_reflections\n");
     }
     if(cudaMemcpy(gpu_reflections, cpu_reflections, sizeof(size_t), cudaMemcpyHostToDevice) != cudaSuccess) {
     fprintf( stderr, "Fail to copy size_t reflections to GPU\n");
     }
    **/

    
    
    int N = WIDTH * HEIGHT;


    // a thread for each pixel
    set_quadrant_color <<<(N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>> (gpu_viewport, gpu_result_array, gpu_scene);
    cudaDeviceSynchronize();


    // copy result array to CPU
    if(cudaMemcpy(cpu_result_array, gpu_result_array, sizeof(vec) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost) != cudaSuccess) {
      fprintf( stderr, "Fail to copy result_array to CPU\n");
    }


    
    // Loop over all pixels in the bitmap
    /*thread q1 = thread(set_quadrant_color, 0, WIDTH/4, view, yrot, result_array);
      thread q2 = thread(set_quadrant_color, WIDTH/4, 2*WIDTH/4, view, yrot,result_array );
      thread q3 = thread(set_quadrant_color, 2*WIDTH/4, 3*WIDTH/4, view, yrot,result_array );
      thread q4 = thread(set_quadrant_color, 3*WIDTH/4, WIDTH, view, yrot,result_array );
    
      q1.join();
      q2.join();
      q3.join();
      q4.join();*/


    // would it be faster to do this inside the kernel, and then copy over the bitmap in the end?
    // instead of writing to an array and copying that back and then running these 2 for loops?
    for (int x = 0 ; x < WIDTH; x++){
      for(int y = 0; y < HEIGHT; y++){
        cpu_bmp.set(x, y, cpu_result_array[x][y]);
      }
    }

    // Display the rendered frame
    ui.display(cpu_bmp);
  }
  
  return 0;
}

// computes the color for the quadrants
__global__ void set_quadrant_color(viewport view, vec* result_array, shape** gpu_scene){
  int index_x = threadIdx.x + blockIdx.x * blockDim.x;
  int index_y = threadIdx.y + blockIdx.y * blockDim.y;
  vec result = raytrace(view.origin(), view.dir(index_x, index_y), 0, gpu_scene);
  // Set the pixel color
  result_array[index_x*WIDTH  + index_y] = result;
}

/**
 * Follow a ray backwards through the scene and return the ray's color
 * \param origin        The origin of the ray
 * \param dir           The direction of the ray
 * \param reflections   The number of times this ray has been reflected
 * \returns             The color of this ray
 */
CUDA_CALLABLE_MEMBER vec raytrace(vec origin, vec dir, size_t reflections, shape** gpu_scene) {
  // Normalize the direction vector
  dir = dir.normalized();
  
  // Keep track of the closest shape that is intersected by this ray
  shape* intersected = NULL;
  float intersect_distance = 0;
  
  // Loop over all shapes in the scene to find the closest intersection
  for(int i = 0; i < OBJ_NUM; i++) {
    float distance = gpu_scene[i]->intersection(origin, dir);
    if(distance >= 0 && (distance < intersect_distance || intersected == NULL)) {
      intersect_distance = distance;
      intersected = gpu_scene[i];
    }
  }
  
  // If the ray didn't intersect anything, just return the ambient color
  if(intersected == NULL) return vec(AMBIENT, AMBIENT, AMBIENT);


  // Without reflections
  
  // Compute the point where the intersection occurred
  vec intersection = origin + dir * intersect_distance;
  
  // Otherwise just return the color of the object
  return intersected->get_color(intersection);

  // With reflections

  /*
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
  return result; */
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
  scene[0] = red_sphere;
  
  // Add a green sphere
  sphere* green_sphere = new sphere(vec(-15, 25, -25), 25);
  green_sphere->set_color(vec(0.125, 0.6, 0.125));
  green_sphere->set_reflectivity(0.5);
  scene[1] = green_sphere;
  
  // Add a blue sphere
  sphere* blue_sphere = new sphere(vec(-50, 40, 75), 40);
  blue_sphere->set_color(vec(0.125, 0.125, 0.75));
  blue_sphere->set_reflectivity(0.5);
  scene[2] = blue_sphere;
  
  // Add a flat surface
  plane* surface = new plane(vec(0, 0, 0), vec(0, 1, 0));
  // The following line uses C++'s lambda expressions to create a function
  /*
  surface->set_color([](vec pos) {
      // This function produces a grid pattern on the plane
      if((int)pos.x() % 100 == 0 || (int)pos.z() % 100 == 0) {
        return vec(0.3, 0.3, 0.3);
      } else {
        return vec(0.15, 0.15, 0.15);
      }
    });
  */ 
  surface->set_diffusion(0.25);
  surface->set_spec_density(10);
  surface->set_spec_intensity(0.1);
  scene[3] = surface;
  
  // Add two lights
  lights[0] = vec(-1000, 300, 0);
  lights[1] = vec(100, 900, 500);
}
