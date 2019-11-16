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

unsigned int output[4000 * 4000 * 3];

unsigned int width, height;
unsigned int imgres[IMG_HEIGHT][IMG_WIDTH];
unsigned int imageMatrix[IMG_HEIGHT][IMG_WIDTH];

unsigned int resultImage[IMG_HEADER + IMG_WIDTH*IMG_HEIGHT];

__device__ unsigned int dev_orig[IMG_HEIGHT][IMG_WIDTH];
__device__ unsigned int dev_result[IMG_HEIGHT][IMG_WIDTH];

void readBMP(char* filename)
{
	int i;
	FILE* f = fopen(filename, "rb");
	unsigned char info[54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

	for (size_t i = 0; i < 54; i++)
	{
		resultImage[i] = info[i];
	}

	// extract image height and width from header
	int width = *(int*)&info[18];
	int height = *(int*)&info[22];

	int size = width * height*3;
	
	fread(output, sizeof(unsigned int), size*3, f); // read the rest of the data at once
	fclose(f);

	for (size_t i = 0; i < IMG_HEIGHT; i++)
	{
		for (size_t j = 0; j < IMG_WIDTH; j++)
		{
			imageMatrix[i][j] = output[i * IMG_HEIGHT + j + 3];
		}
	}
}


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
			int curPixel = dev_orig[row + j][col + i];
			sumX += curPixel * Gx[i + 1][j + 1];
			sumY += curPixel * Gy[i + 1][j + 1];
		}
	}

	int sum = abs(sumY) + abs(sumX);
	if (sum > 255) sum = 255;
	if (sum < 0) sum = 0;

	dev_result[row][col] = dev_orig[row][col];
}


int main()
{
	printf("Starting program \n\n");

	FILE * f_output_img;

	readBMP(IMG_INPUT);

	//load image to CUDA managed memory
	cudaMemcpyToSymbol(dev_orig, imageMatrix, sizeof(unsigned int) * IMG_WIDTH * IMG_HEIGHT);

	CPreciseTimer timer;
	for (int _timeri = 0; _timeri <= 1; _timeri++) {
		if (_timeri == 1)
			timer.StartTimer();
		EdgeDetect << <dim3(125, 125), dim3(32, 32) >> > (IMG_WIDTH, IMG_HEIGHT);
	}
	timer.StopTimer();

	printf("Timer[%s]=%f\n\n", "Edge decetion", (float)timer.GetTimeMilliSec() / 1);

	cudaMemcpyFromSymbol(imgres, dev_result, sizeof(unsigned int) * IMG_WIDTH * IMG_HEIGHT);

	for (size_t i = 0; i < IMG_HEIGHT; i++)
	{
		for (size_t j = 0; j < IMG_WIDTH; j++)
		{
			resultImage[(i * 4000 + j) + 54] = imgres[i][j];
		}
	}
	

	// Save file
	fopen_s(&f_output_img, IMG_OUTPUT2, "wb");
	fwrite(resultImage, 1, IMG_HEADER + IMG_WIDTH * IMG_HEIGHT, f_output_img);
	fclose(f_output_img);

	// Free CUDA managed host memory

	printf("Ending program \n");
}
