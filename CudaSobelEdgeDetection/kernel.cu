#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <malloc.h>
#include <stdio.h>
#include "res\Timing.h"

// Image data
// baby 1080; 4000; 4000; 125; 125; 32; 32
// oldlady 1080; 512; 512; 16; 16; 32; 32
// lena 1080; 512; 512; 16; 16; 32; 32

#define IMG_INPUT "img\\lena.bmp"
#define IMG_OUTPUT "img\\output.bmp"

#define IMG_HEADER 1080
#define IMG_WIDTH 512
#define IMG_HEIGHT 512

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16
#define THREAD_X 32
#define THREAD_Y 32

//minden block 2 dimenziós lesz, 32*32 így minden block 1024 szálat tud futtatni egy időben
//ez annyit jelent, hogy egy 4000*4000-es képnél 125 block vízszintesen, 125 függőlegesen


__global__ void SobelEdgeDetection(unsigned char* img)
{
	//ez felel meg a szekvenciális kódban a két egybeágyazott for ciklusnak
	int row = blockIdx.y * blockDim.y + threadIdx.y; //i blockidx a hanyadik block az oszlopban, blockdim az a blokkon belüli sor
	int col = blockIdx.x * blockDim.x + threadIdx.x; //j

	if (row >= IMG_HEIGHT || col >= IMG_WIDTH || row < 1 || col < 1) return; //olyan szál le se fusson ami kiindexelne a képből

	int Gx[3][3] = { {-1,0,1}, {-2,0,2}, {-1,0,1} };
	int Gy[3][3] = { {1,2,1}, {0,0,0}, {-1,-2,-1} };

	int sumX = 0;
	int sumY = 0;

	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			char curPixel = img[(row + j) * IMG_WIDTH + (col + i)];
			sumX += (curPixel + '0') * Gx[i + 1][j + 1];
			sumY += (curPixel + '0') * Gy[i + 1][j + 1];
		}
	}

	int sum = sumY + sumX;
	if (sum > 255) 
		sum = 255;
	if (sum < 0) 
		sum = 0;

	__syncthreads();

	img[row * IMG_WIDTH + col] = sum - '0';
}


void EdgeDetectionCaller(unsigned char* img)
{
	//Allocate device memory
	unsigned char* d_img;
	cudaMalloc((void**)&d_img, sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT);


	//Memory copy H->D
	cudaMemcpy(d_img, img + IMG_HEADER, sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT, cudaMemcpyHostToDevice);


	//Launch kernel
	SobelEdgeDetection << < dim3(BLOCKSIZE_X, BLOCKSIZE_Y), dim3(THREAD_X, THREAD_Y) >> > (d_img);


	//Memory copy D->H
	cudaMemcpy(img + IMG_HEADER, d_img, sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToHost);


	//Free device memory
	cudaFree(d_img);

}

int main()
{
	unsigned char* img;
	FILE* f_input_img, * f_output_img;

	// Load image
	img = (unsigned char*)malloc(IMG_HEADER + sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT);

	fopen_s(&f_input_img, IMG_INPUT, "rb");
	fread(img, 1, IMG_HEADER + IMG_WIDTH * IMG_HEIGHT, f_input_img);
	fclose(f_input_img);

	// Run single stream kernel
	MEASURE_TIME(1, "ReduceContrastDefaultStream", EdgeDetectionCaller(img));

	// Save file
	fopen_s(&f_output_img, IMG_OUTPUT, "wb");
	fwrite(img, 1, IMG_HEADER + IMG_WIDTH * IMG_HEIGHT, f_output_img);
	fclose(f_output_img);
	free(img);
}
