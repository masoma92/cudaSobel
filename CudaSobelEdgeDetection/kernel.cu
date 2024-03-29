﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include "res\Timing.h"

/** HELPER
** Image data
** baby 1080; 4000; 4000; 125; 125; 32; 32
** lady 1080; 512; 512; 16; 16; 32; 32
** lena 1080; 512; 512; 16; 16; 32; 32 
**/

#define IMG_INPUT "img\\baby.bmp"
#define IMG_OUTPUT "img\\outputSingle.bmp"
#define IMG_OUTPUT2 "img\\outputMultiple.bmp"

#define IMG_HEADER 1080
#define IMG_WIDTH 4000
#define IMG_HEIGHT 4000

#define BLOCKSIZE_X 512
#define BLOCKSIZE_Y 512
#define THREAD_X 32
#define THREAD_Y 32

// minden block 2 dimenziós, 32*32 szál/block -> 1024 szál egy időben
// ez egy 4000*4000-es képnél 125 block vízszintesen, 125 függőlegesen


__global__ void EdgeDetectionKernel(unsigned char* input_img, unsigned char* output_img)
{
	// Ez felel meg a szekvenciális kódban a két egybeágyazott for ciklusnak
	int row = blockIdx.y * blockDim.y + threadIdx.y; //i blockidx a hanyadik block a az x/y tengelyen, blockdim -> 32 x/y tengely blockon belül
	int col = blockIdx.x * blockDim.x + threadIdx.x; //j

	// Szűrő lsd.: documentation_hun.pdf
	int Gx[3][3] = { {-1,0,1}, {-2,0,2}, {-1,0,1} };
	int Gy[3][3] = { {1,2,1}, {0,0,0}, {-1,-2,-1} };

	double sumX = 0;
	double sumY = 0;
	double sum = 0;

	// Szűrő alkalmazása az adott blockon belüli képrészletre
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {

			char curPixel = input_img[(row * IMG_HEIGHT + col) + (i * IMG_WIDTH + j)];
			sumX += (curPixel) * Gx[i + 1][j + 1];
			sumY += (curPixel) * Gy[i + 1][j + 1];
		}
	}

	sum = sqrt(pow(sumX, 2) + pow(sumY, 2));
	if (sum > 255) 
		sum = 255;

	output_img[row * IMG_WIDTH + col] = sum;
}

void EdgeDetectionCaller(unsigned char* img, unsigned char* img_output)
{
	unsigned char* d_img;
	unsigned char* d_img_output;

	// Gpu memória allokáció
	cudaMalloc((void**)&d_img, sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT);
	cudaMalloc((void**)&d_img_output, sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT);

	// Másolás H->D
	cudaMemcpy(d_img, img + IMG_HEADER, sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT, cudaMemcpyHostToDevice);


	// Kernel futtatás
	EdgeDetectionKernel << < dim3(BLOCKSIZE_X, BLOCKSIZE_Y), dim3(THREAD_X, THREAD_Y) >> > (d_img, d_img_output);


	// Másolás D->H
	cudaMemcpy(img + IMG_HEADER, d_img, sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToHost);
	cudaMemcpy(img_output + IMG_HEADER, d_img_output, sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToHost);


	// Memória felszabadítás
	cudaFree(d_img);
	cudaFree(d_img_output);
}

void EdgeDetectionSequential(unsigned char* input_img, unsigned char* output_img) {

	int Gx[3][3] = { {-1,0,1}, {-2,0,2}, {-1,0,1} };
	int Gy[3][3] = { {1,2,1}, {0,0,0}, {-1,-2,-1} };

	double sumX = 0;
	double sumY = 0;
	double sum = 0;

	for (int i = 1; i < IMG_HEIGHT-1; i++) 
	{
		for (int j = 1; j < IMG_WIDTH-1; j++)
		{
			for (int k = -1; k <= 1; k++) {
				for (int l = -1; l <= 1; l++) {

					char curPixel = input_img[(i * IMG_HEIGHT + j) + (k * IMG_WIDTH + l)];

					sumX += (curPixel) * Gx[k + 1][l + 1];
					sumY += (curPixel) * Gy[k + 1][l + 1];
				}
			}

			sum = sqrt(pow(sumX, 2) + pow(sumY, 2));

			if (sum > 255)
				sum = 255;

			output_img[i * IMG_WIDTH + j] = sum;

			sumX = 0;
			sumY = 0;
			sum = 0;
		}
	}
}

int main()
{
	// 0 - 255
	unsigned char* img;
	unsigned char* img_output;
	FILE* f_input_img, * f_output_img;

	// Kép betöltése
	img = (unsigned char*)malloc(IMG_HEADER + sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT);
	img_output = (unsigned char*)malloc(IMG_HEADER + sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT);

	fopen_s(&f_input_img, IMG_INPUT, "rb");
	fread(img, 1, IMG_HEADER + IMG_WIDTH * IMG_HEIGHT, f_input_img);
	fclose(f_input_img);

	fopen_s(&f_input_img, IMG_INPUT, "rb");
	fread(img_output, 1, IMG_HEADER + IMG_WIDTH * IMG_HEIGHT, f_input_img);
	fclose(f_input_img);

	// Szekvenciális megoldás futási idejének mérése
	MEASURE_TIME(1, "EdgeDetectionSequential", EdgeDetectionSequential(img + IMG_HEADER, img_output + IMG_HEADER));

	fopen_s(&f_output_img, IMG_OUTPUT, "wb");
	fwrite(img_output, 1, IMG_HEADER + IMG_WIDTH * IMG_HEIGHT, f_output_img);
	fclose(f_output_img);

	// Cuda kernel megoldás futási idejének mérése
	MEASURE_TIME(1, "EdgeDetectionParallelKernel", EdgeDetectionCaller(img, img_output));

	// Fájl mentése
	fopen_s(&f_output_img, IMG_OUTPUT2, "wb");
	fwrite(img_output, 1, IMG_HEADER + IMG_WIDTH * IMG_HEIGHT, f_output_img);
	fclose(f_output_img);
	free(img);
	free(img_output);
}
