# csc213-GPURayTracing
Final project
This is the product of collaboration among four members in the group : Medha Gopalaswamy, Khoa Nguyen, Ameer Shujjah, and Oanh Vu.

## Overview
- Extended the [rayTracer lab](http://www.cs.grinnell.edu/~curtsinger/teaching/2016S/CSC213/labs/raytracer/)
- Pixel computation is now done on the GPUs, rather than on CPU
- Pixel color is then copied it back and displayed on CPU

## Design and Implementation
- Oversampling
- Animation
- Reflection 
- Lighting 
- Highlight

## Challenges
- CUDA can’t handle abstracts objects or recursion
- We only knew very basic CUDA functionality
- Certain functions of objects (such as vector, bitmap) did not work as expected in CUDA
- Hard to locate source of errors in sections of the code performed on GPUs
- Makefile: GPU and SDL

## What we accomplished:
- Learned about the setup in depth
- Re-structured classes to work with CUDA
- Work around C++ polymorphism
- Troubleshooting CUDA memory errors


