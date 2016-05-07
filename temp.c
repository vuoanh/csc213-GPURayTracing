CUDA_CALLABLE_MEMBER vec raytrace(vec origin, vec dir, size_t reflections,
                                  sphere* gpu_spheres, plane* gpu_plane) {
  
  // Keep track of the closest shape that is intersected by this ray
  int intersected = 0;
  float intersect_distance = 0;
  plane current_plane;
  sphere current_sphere;
  int plane_closer = 0;
  int sphere_index = 0;
  vec final_result;

  while(reflection < MAX_REFLECTIONS) {
  
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
  
    // for the plane
    current_plane = *gpu_plane;
    float distance = current_plane.intersection(origin, dir);
    if(distance >= 0 && (distance < intersect_distance || !intersected)) {
      intersected = 2;
      intersect_distance = distance;
      plane_closer = 1;
      //intersected = &current_plane;
    }

    if(!intersected)
      return vec(AMBIENT, AMBIENT, AMBIENT);

    origin  = origin + dir * (intersect_distance - EPSILON);

    vec n;
    vec result;
  
    // Initialize the result color to the ambient light reflected in the shapes color
    if(intersected == 1) {
      result = gpu_spheres[sphere_index].get_color(origin) * AMBIENT;
    }
    else {
      result = current_plane.get_color(origin) * AMBIENT;
    }

    // Find the normal at the intersection point
    if(intersected == 1) {
      n = gpu_spheres[sphere_index].normal(origin);
    }
    else {
      n = current_plane.normal(origin);
    }

    // Reflect the vector across the normal
    dir = dir - n * 2.0 * n.dot(dir);
      
    // Add the reflection to the result, tinted by the color of the shape
    if(intersected == 1) {
      final_result += result.hadamard(gpu_spheres[sphere_index].get_color(origin)) *
        gpu_spheres[sphere_index].get_reflectivity();
    }
    else {
      final_result += result.hadamard(current_plane.get_color(origin)) *
        current_plane.get_reflectivity();
    }
    
  }
  
  return final_result;
}
