#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include "res\Timing.h"

// Image data
// baby 1080; 4000; 4000; 125; 125; 32; 32
// lady 1080; 512; 512; 16; 16; 32; 32
// lena 1080; 512; 512; 16; 16; 32; 32

#define IMG_INPUT "img\\lena.bmp"
#define IMG_OUTPUT "img\\outputSingle.bmp"
#define IMG_OUTPUT2 "img\\outputMultiple.bmp"

#define IMG_HEADER 1080
#define IMG_WIDTH 512
#define IMG_HEIGHT 512

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16
#define THREAD_X 32
#define THREAD_Y 32

//minden block 2 dimenziós lesz, 32*32 így minden block 1024 szálat tud futtatni egy időben
//ez annyit jelent, hogy egy 4000*4000-es képnél 125 block vízszintesen, 125 függőlegesen


__global__ void EdgeDetectionKernel(unsigned char* img, unsigned char* img_output)
{
	//ez felel meg a szekvenciális kódban a két egybeágyazott for ciklusnak
	int row = blockIdx.y * blockDim.y + threadIdx.y; //i blockidx a hanyadik block az oszlopban, blockdim.y = 32 thread db szám vízszintesen
	int col = blockIdx.x * blockDim.x + threadIdx.x; //j

	if (row >= IMG_HEIGHT || col >= IMG_WIDTH || row < 1 || col < 1) return; //olyan szál le se fusson ami kiindexelne a képből

	int Gx[3][3] = { {-1,0,1}, {-2,0,2}, {-1,0,1} };
	int Gy[3][3] = { {1,2,1}, {0,0,0}, {-1,-2,-1} };

	double sumX = 0;
	double sumY = 0;
	double sum = 0;

	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			char curPixel = img[(row + j) * IMG_WIDTH + (col + i)];
			sumX += (curPixel) * Gx[i + 1][j + 1];
			sumY += (curPixel) * Gy[i + 1][j + 1];
		}
	}

	
	sum = sqrt(pow(sumY, 2) + pow(sumX, 2));
	if (sum > 255) 
		sum = 255;
	if (sum < 0) 
		sum = 0;

	img_output[row * IMG_WIDTH + col] = sum;
}

void EdgeDetectionCaller(unsigned char* img, unsigned char* img_output)
{
	//Allocate device memory
	unsigned char* d_img;
	unsigned char* d_img_output;

	cudaMalloc((void**)&d_img, sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT);
	cudaMalloc((void**)&d_img_output, sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT);


	//Memory copy H->D
	cudaMemcpy(d_img, img + IMG_HEADER, sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT, cudaMemcpyHostToDevice);


	//Launch kernel
	EdgeDetectionKernel << < dim3(BLOCKSIZE_X, BLOCKSIZE_Y), dim3(THREAD_X, THREAD_Y) >> > (d_img, d_img_output);


	//Memory copy D->H
	cudaMemcpy(img + IMG_HEADER, d_img, sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToHost);
	cudaMemcpy(img_output + IMG_HEADER, d_img_output, sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToHost);


	//Free device memory
	cudaFree(d_img);
	cudaFree(d_img_output);
}

void EdgeDetectionSequential(unsigned char* input_img, unsigned char* output_img) {

	int Gx[3][3] = { {-1,0,1}, {-2,0,2}, {-1,0,1} };
	int Gy[3][3] = { {1,2,1}, {0,0,0}, {-1,-2,-1} };

	int sumX = 0;
	int sumY = 0;
	int sum = 0;

	for (int i = IMG_HEADER + IMG_WIDTH; i < (IMG_HEIGHT*IMG_WIDTH)-IMG_WIDTH; i++) //első és utolsó sornyi pixelt kihagyja
	{
		if (((i - IMG_HEADER) % (IMG_WIDTH)) != 0 && ((i - IMG_HEADER) % (IMG_WIDTH)) != IMG_WIDTH-1) { //széleket kihagyja

			for (int k = -1; k <= 1; k++) {
				for (int l = -1; l <= 1; l++) {

					char curPixel = input_img[i + (k * IMG_WIDTH) + l];
					sumX += (curPixel) * Gx[k + 1][l + 1];
					sumY += (curPixel) * Gy[k + 1][l + 1];
				}
			}
		}

		sum = sqrt(pow(sumY, 2) + pow(sumX, 2));

		if (sum > 255)
			sum = 255;
		if (sum < 0)
			sum = 0;

		output_img[i] = sum;
	}

	
}

int main()
{
	unsigned char* img;
	unsigned char* img_output;
	FILE* f_input_img, * f_output_img;

	// Load image
	img = (unsigned char*)malloc(IMG_HEADER + sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT);
	img_output = (unsigned char*)malloc(IMG_HEADER + sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT);

	fopen_s(&f_input_img, IMG_INPUT, "rb");
	fread(img, 1, IMG_HEADER + IMG_WIDTH * IMG_HEIGHT, f_input_img);
	fclose(f_input_img);

	fopen_s(&f_input_img, IMG_INPUT, "rb");
	fread(img_output, 1, IMG_HEADER + IMG_WIDTH * IMG_HEIGHT, f_input_img);
	fclose(f_input_img);

	MEASURE_TIME(1, "EdgeDetectionSequential", EdgeDetectionSequential(img, img_output));

	fopen_s(&f_output_img, IMG_OUTPUT, "wb");
	fwrite(img_output, 1, IMG_HEADER + IMG_WIDTH * IMG_HEIGHT, f_output_img);
	fclose(f_output_img);

	// Run cuda kernel
	MEASURE_TIME(1, "EdgeDetectionParallelKernel", EdgeDetectionCaller(img, img_output));

	// Save file
	fopen_s(&f_output_img, IMG_OUTPUT2, "wb");
	fwrite(img_output, 1, IMG_HEADER + IMG_WIDTH * IMG_HEIGHT, f_output_img);
	fclose(f_output_img);
	free(img);
	free(img_output);
}
