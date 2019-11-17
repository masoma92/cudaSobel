#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include "res\Timing.h"

#define IMG_INPUT "img\\input.bmp"
#define IMG_OUTPUT "img\\output.bmp"
#define IMG_OUTPUT2 "img\\output2.bmp"
#define IMG_HEADER 1078
#define IMG_WIDTH 4000
#define IMG_HEIGHT 4000

//minden block 2 dimenziós lesz, 32*32 így minden block 1024 szálat tud futtatni egy időben
//ez annyit jelent, hogy egy 4000*4000-es képnél 125 block vízszintesen, 125 függőlegesen

__device__ char* dev_origin;
__device__ char* dev_result;

__global__ void EdgeDetect(int width, int height) {

	//ez felel meg a szekvenciális kódban a két egybeágyazott for ciklusnak
	int row = blockIdx.y * blockDim.y + threadIdx.y; //i blockidx a hanyadik block az oszlopban, blockdim az a blokkon belüli sor
	int col = blockIdx.x * blockDim.x + threadIdx.x; //j


	if (row >= height || col >= width || row < 1 || col < 1) return; //olyan szál le se fusson ami nem a képen kívül van

	int Gx[3][3] = { {-1,0,1}, {-2,0,2}, {-1,0,1} };
	int Gy[3][3] = { {1,2,1}, {0,0,0}, {-1,-2,-1} };

	int sumX, sumY;
	sumX = sumY = 0;

	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			int curPixel = dev_origin[(row + j) * width + (col + i)];
			sumX += curPixel * Gx[i + 1][j + 1];
			sumY += curPixel * Gy[i + 1][j + 1];
		}
	}

	int sum = abs(sumY) + abs(sumX);
	if (sum > 255) sum = 255;
	if (sum < 0) sum = 0;

	dev_result[row * width + col] = sum;
}


int main()
{
	FILE* f_input_img, * f_output_img;
	unsigned char* img;
	unsigned char* out_img;

	// Allocate CUDA managed host memory
	cudaMallocHost((void**)&img, IMG_HEADER + sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT, 0);
	cudaMallocHost((void**)&out_img, IMG_HEADER + sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT, 0);

	// Load header
	fopen_s(&f_input_img, IMG_INPUT, "rb");
	fread(img, 1, IMG_HEADER, f_input_img);

	//ToDo: Allocate device memory

	cudaMalloc((void**)&img, IMG_HEADER + sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT);
	cudaMemcpy((void*)&dev_origin, &img, IMG_HEADER + sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT, cudaMemcpyHostToDevice);

	cudaMemcpy((void**)&dev_result, &img, IMG_HEADER + sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT, cudaMemcpyHostToDevice);

	EdgeDetect << <dim3(125 * 125), dim3(32 * 32) >> > (IMG_WIDTH, IMG_HEIGHT);

	cudaMemcpy((void**)&img, &dev_result, IMG_HEADER + sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToHost);

		// Save file
	fopen_s(&f_output_img, IMG_OUTPUT, "wb");
	fwrite(img, 1, IMG_HEADER + IMG_WIDTH * IMG_HEIGHT, f_output_img);
	fclose(f_output_img);

	//ToDo: Free device memory

		// Free CUDA managed host memory
	cudaFreeHost(img);
}
