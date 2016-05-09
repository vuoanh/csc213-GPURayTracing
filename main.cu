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

// http://stackoverflow.com/questions/6978643/cuda-and-classes
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

// CUDA error checking from http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-ap
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Screen size
#define WIDTH 640
#define HEIGHT 480
#define N WIDTH*HEIGHT

// Rendering Properties
#define AMBIENT 0.3         // Ambient illumination
#define OVERSAMPLE 2       // Sample 2x2 subpixels
#define MAX_REFLECTIONS 1  // The maximum number of times a ray is reflected
#define EPSILON 0.03        // Shift points off surfaces by this much
// Create threads for oversampling
#define num_over_thread OVERSAMPLE*OVERSAMPLE
#define THREADS_PER_BLOCK 8
#define OBJ_NUM 3
#define LIGHT_NUM 2
using namespace std;

// Set up the 3D scene
void init_scene();

// Trace a ray through the scene to determine its color
CUDA_CALLABLE_MEMBER vec raytrace(vec origin, vec dir, size_t reflections, sphere* gpu_spheres,
                                  plane* gpu_plane, vec* gpu_lights);

// A list of spheres that make up the 3D scene. Initialized by init_scene
sphere scene[OBJ_NUM];

// A plane
plane cpu_plane;

// A list of light positions, all emitting pure white light
vec lights[LIGHT_NUM];

// computes the color for the pixels
__global__ void set_pixel_color(viewport* view, sphere* gpu_scene, plane* gpu_plane,
                                   bitmap* gpu_bmp, float* gpu_yrot, vec* gpu_lights);

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

  // initiates and allocate memory for GPU shapes
  sphere* gpu_spheres;
  if (cudaMalloc(&gpu_spheres, sizeof(sphere) * OBJ_NUM) != cudaSuccess) {
    fprintf( stderr, "Fail to allocate GPU objects\n");
  }
  if(cudaMemcpy(gpu_spheres, scene, sizeof(sphere) * OBJ_NUM, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf( stderr, "Fail to copy objects to GPU\n");
  }

  // initiates and allocate memory for GPU plane
  plane* gpu_plane;
  if (cudaMalloc(&gpu_plane, sizeof(plane)) != cudaSuccess) {
    fprintf( stderr, "Fail to allocate GPU plane\n");
  }
  if(cudaMemcpy(gpu_plane, &cpu_plane, sizeof(plane), cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf( stderr, "Fail to copy plane to GPU\n");
  }
  
    
  // initiates and allocate memory for GPU lights
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
    float yrot = (time_ms() - start_time)/4000.0 * M_PI * 2;
    float* gpu_yrot;
    
    // Allocate memory for GPU y-rotation
    if (cudaMalloc(&gpu_yrot, sizeof(float))!= cudaSuccess) {
      fprintf( stderr, "Fail to allocate GPU yrot\n");
    }
    if(cudaMemcpy(gpu_yrot, &yrot, sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf( stderr, "Fail to copy yrot to GPU\n");
    }
    
    // Render the frame to this bitmap
    bitmap cpu_bmp;
    bitmap* gpu_bmp;
  
    // Allocate memory for the gpu bitmap and the gpu result array
    gpuErrchk(cudaMalloc(&gpu_bmp, cpu_bmp.size()));

    // Copy memory from the cpu bitmap and result array to the gpu counterparts
    if(cudaMemcpy(gpu_bmp, &cpu_bmp, cpu_bmp.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf( stderr, "Fail to copy bitmap to GPU\n");
    }

    // Allocate memory for gpu viewport
    viewport* gpu_viewport;
    if (cudaMalloc(&gpu_viewport, sizeof(viewport))!= cudaSuccess) {
      fprintf( stderr, "Fail to allocate GPU viewport\n");
    }
    if(cudaMemcpy(gpu_viewport, &view, sizeof(viewport), cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf( stderr, "Fail to copy viewport to GPU\n");
    }

      // Create a thread for each pixel
    set_pixel_color <<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
      THREADS_PER_BLOCK>>> (gpu_viewport, gpu_spheres, gpu_plane, gpu_bmp, gpu_yrot, gpu_lights);
      
     // Check for any errors in gpu
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
     // Copy result array to CPU
    if(cudaMemcpy(&cpu_bmp, gpu_bmp, cpu_bmp.size(), cudaMemcpyDeviceToHost)
       != cudaSuccess) {
      fprintf( stderr, "Fail to copy result_array to CPU\n");
    }
    // Display the rendered frame
    ui.display(cpu_bmp);

  // Free variables that are reallocated within while-loop
    cudaFree(gpu_bmp);
    cudaFree(gpu_viewport);
    cudaFree(gpu_yrot);
  }

// Free all other variables
  cudaFree(gpu_spheres);
  cudaFree(gpu_plane);
  cudaFree(gpu_lights);
  
  return 0;
}

// computes the color for the quadrants
__global__ void set_pixel_color(viewport* view, sphere* gpu_spheres, plane* gpu_plane, bitmap* gpu_bmp,  float* gpu_yrot, vec* gpu_lights){
  
  // Calculate the y and x indices from thread, block id
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int index_x = index % WIDTH;
  int index_y = index / WIDTH;
  
  vec result;
  for(int y_sample = 0; y_sample < OVERSAMPLE; y_sample++) {
    // The y offset is half way between the edges of this subpixel
    float y_off = (y_sample + 0.5) / OVERSAMPLE;
          
    // Loop over x subpixel positions
    for(int x_sample = 0; x_sample < OVERSAMPLE; x_sample++) {
      // The x offset is half way between the edges of this subpixel
      float x_off = (x_sample + 0.5) / OVERSAMPLE;
      
      // Raytrace from the viewport origin through the viewing 
      result += raytrace(view->origin().yrotated(*gpu_yrot), view->dir(index_x + x_off, index_y + y_off).yrotated(*gpu_yrot), 0, gpu_spheres, gpu_plane, gpu_lights);
    }
  }
  // Set the pixel color
  result /= OVERSAMPLE*OVERSAMPLE;
  gpu_bmp->set(index_x, index_y, result);

}

// Trace a ray through the scene to determine its color
CUDA_CALLABLE_MEMBER vec raytrace(vec origin, vec dir, size_t reflections,
                                  sphere* gpu_spheres, plane* gpu_plane, vec* gpu_lights) {
  
  // Keep track of the closest shape that is intersected by this ray
  plane current_plane;
  sphere current_sphere;
  vec final_result;
  vec intersection = origin;
  vec current_light;
  
  // Iterate while there are reflections left to do
  while(reflections < MAX_REFLECTIONS) {
    int plane_closer = 0;
    int sphere_index = 0;
    int intersected = 0;
    float intersect_distance = 0;
    origin = intersection;
  
    // Normalize the direction vector
    dir = dir.normalized();
  
    // Loop over all spheres in the scene to find the closest intersection
    for(int i = 0; i < OBJ_NUM ; i++) {
      current_sphere = gpu_spheres[i];
      float distance = current_sphere.intersection(origin, dir);
        if(distance >= 0 && (distance < intersect_distance || !intersected)) {
        intersect_distance = distance;
        intersected = 1;
        sphere_index = i;
      }
    }
  
    // Loop over the plane in the scene to find the closest intersection
    current_plane = *gpu_plane;
    float distance = current_plane.intersection(origin, dir);
    if(distance >= 0 && (distance < intersect_distance || !intersected)) {
      intersected = 1;
      intersect_distance = distance;
      plane_closer = 1;
    }
   // if not intersected to any shapes
    if(intersected == 0)
      return vec(AMBIENT, AMBIENT, AMBIENT);

    intersection = origin + dir * (intersect_distance - EPSILON);

    
    vec n; // normalized vector
    vec result; //the result color 
    
    // Initialize the result color to the ambient light reflected in the shapes color
    if(!plane_closer) {
      result = gpu_spheres[sphere_index].get_color(intersection) * AMBIENT;
      n = (intersection - gpu_spheres[sphere_index].get_pos()).normalized();
    }
    else {
      result = current_plane.get_color(intersection) * AMBIENT;
      n = current_plane.get_norm();
    }
  
    // Reflect the vector across the normal
    dir = dir - n * 2.0 * n.dot(dir);
      
    // Add the reflection to the result, tinted by the color of the shape
    if(!plane_closer) {
      final_result += result.hadamard(gpu_spheres[sphere_index].get_color(intersection)) *
        gpu_spheres[sphere_index].get_reflectivity();
    }
    else {
      final_result += result.hadamard(current_plane.get_color(intersection)) *
        current_plane.get_reflectivity();
    }
    // For the lights
    for(int i=0; i < LIGHT_NUM; i++) {
      current_light = gpu_lights[i];
      bool in_shadow = false;
      // Create a unit vector from the intersection to the light source
      vec shadow_dir = (current_light - intersection).normalized();

      // for the spheres
      for(int i = 0; i < OBJ_NUM ; i++) {
        current_sphere = gpu_spheres[i];
        if(current_sphere.intersection(intersection, shadow_dir) >= 0) {
          in_shadow = true;
          break;
        }
      }

      // for the plane
      current_plane = *gpu_plane;
      if(current_plane.intersection(intersection, shadow_dir) >= 0) {
        in_shadow = true;
      }
      
      
      // If there is a clear path to the light, add illumination
      if(!in_shadow) {
        vec bisector = (shadow_dir - dir).normalized();
        float specular_intensity;
        if(!plane_closer) {
          float diffuse_intensity = gpu_spheres[sphere_index].get_diffusion() * fmax(0, n.dot(shadow_dir));     
          // Add diffuse lighting tinted by the color of the shape
          final_result += gpu_spheres[sphere_index].get_color(intersection) * diffuse_intensity;
          specular_intensity = gpu_spheres[sphere_index].get_spec_intensity() *
            fmax(0, pow(n.dot(bisector), (int)gpu_spheres[sphere_index].get_spec_density()));
        }
        else {
          float diffuse_intensity = current_plane.get_diffusion() * fmax(0, n.dot(shadow_dir));     
          // Add diffuse lighting tinted by the color of the shape
          final_result += current_plane.get_color(intersection) * diffuse_intensity;
          specular_intensity = current_plane.get_spec_intensity() *
            fmax(0, pow(n.dot(bisector), (int)current_plane.get_spec_density()));
        }
      
        // Add specular highlights
        result += vec(1.0, 1.0, 1.0) * specular_intensity;
        
      }
    }
    
    // Increment reflections
    reflections++;
    
  }
  
  return final_result;
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
  scene[0] = *red_sphere;
  
  // Add a green sphere
  sphere* green_sphere = new sphere(vec(-15, 25, -25), 25);
  green_sphere->set_color(vec(0.125, 0.6, 0.125));
  green_sphere->set_reflectivity(0.5);
  scene[2] = *green_sphere;
  
  // Add a blue sphere
  sphere* blue_sphere = new sphere(vec(-50, 40, 75), 40);
  blue_sphere->set_color(vec(0.125, 0.125, 0.75));
  blue_sphere->set_reflectivity(0.5);
  scene[1] = *blue_sphere;
  
  // Add a flat surface
   plane* surface = new plane(vec(0, 0, 0), vec(0, 1, 0));
  surface->set_diffusion(0.25);
  surface->set_spec_density(10);
  surface->set_spec_intensity(0.1);
  surface->set_color(vec(0.3,0.3,0.3));
  cpu_plane = *surface;

  // Add two lights
  lights[0] = vec(-1000, 300, 0);
  lights[1] = vec(100, 900, 500);
}
