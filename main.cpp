#define _CRT_SECURE_NO_WARNINGS // for stb

// Parameters
#define DETERMINISTIC() false

static const int   c_2DImageSize = 256;
static const int   c_2DNumGaussians = 25;
static const float c_2DSigmaMin = 0.05f;
static const float c_2DSigmaMax = 0.2f;
static const int   c_2DNumSteps = 10;

#include <stdio.h>
#include <array>
#include <random>
#include <vector>
#include <direct.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"


template <size_t N>
using Vec = std::array<float, N>;

using float2 = Vec<2>;

template <size_t N>
Vec<N> operator - (const Vec<N>& A, const Vec<N>& B)
{
	Vec<N> ret;
	for (size_t i = 0; i < N; ++i)
		ret[i] = A[i] - B[i];
	return ret;
}

template <typename T>
inline T Clamp(T value, T themin, T themax)
{
	if (value <= themin)
		return themin;

	if (value >= themax)
		return themax;

	return value;
}

struct Gauss2D
{
	float2 center;
	float2 sigma;
};

float Gauss(float x, float sigma)
{
	return std::exp(-(x * x) / (2.0f * sigma * sigma));
}

template <size_t N>
float Gauss(const Vec<N>& x, const Vec<N>& sigma)
{
	float ret = 1.0f;
	for (size_t i = 0; i < N; ++i)
		ret *= Gauss(x[i], sigma[i]);
	return ret;
}

std::mt19937 GetRNG(unsigned int& seed)
{
#if !DETERMINISTIC()
	std::random_device rd;
	seed = rd();
#endif
	return std::mt19937(seed);
}

void DrawGaussians(const std::vector<Gauss2D>& gaussians, const std::vector<float2>& points, const char* fileName)
{
	// Gather up the gaussian pixel values
	std::vector<float> pixelsF(c_2DImageSize * c_2DImageSize, 0.0f);
	float* pixelF = pixelsF.data();
	float maxValue = 0.0f;
	for (int i = 0; i < c_2DImageSize * c_2DImageSize; ++i)
	{
		int x = i % c_2DImageSize;
		int y = i / c_2DImageSize;

		float2 uv = float2{
			(float(x) + 0.5f) / float(c_2DImageSize),
			(float(y) + 0.5f) / float(c_2DImageSize)
		};

		for (const Gauss2D& gaussian : gaussians)
			*pixelF += Gauss(gaussian.center - uv, gaussian.sigma);

		maxValue = std::max(maxValue, *pixelF);

		pixelF++;
	}

	// normalize and convert to U8
	std::vector<unsigned char> pixels(c_2DImageSize * c_2DImageSize * 3, 0);
	for (size_t i = 0; i < c_2DImageSize * c_2DImageSize; ++i)
	{
		unsigned char U8 = (unsigned char)Clamp(255.0f * pixelsF[i] / maxValue, 0.0f, 255.0f);
		pixels[i * 3 + 0] = U8;
		pixels[i * 3 + 1] = U8;
		pixels[i * 3 + 2] = U8;
	}

	// draw the points
	for (const float2& point : points)
	{
		int px = Clamp(int(point[0] * c_2DImageSize), 0, c_2DImageSize - 1);
		int py = Clamp(int(point[1] * c_2DImageSize), 0, c_2DImageSize - 1);
		pixels[(py * c_2DImageSize + px) * 3 + 0] = 255;
		pixels[(py * c_2DImageSize + px) * 3 + 1] = 255;
		pixels[(py * c_2DImageSize + px) * 3 + 2] = 0;
	}

	stbi_write_png(fileName, c_2DImageSize, c_2DImageSize, 3, pixels.data(), 0);
}

void DoTest2D(const char* baseFileName)
{
	unsigned int seed = 1337;
	std::mt19937 rng = GetRNG(seed);

	// Generate the randomized gaussians
	std::uniform_real_distribution<float> distPos(0.0f, 1.0f);
	std::uniform_real_distribution<float> distSigma(c_2DSigmaMin, c_2DSigmaMax);
	std::vector<Gauss2D> gaussians(c_2DNumGaussians);
	for (Gauss2D& gaussian : gaussians)
	{
		gaussian.center = float2{ distPos(rng), distPos(rng) };
		gaussian.sigma = float2{ distSigma(rng), distSigma(rng) };
	}

	// TODO: show points of gradient descent for a couple different learning rates, and for adam,
	// TODO: make a CSV of movement length each step, to graph it?
	// TODO: report only c_2DNumProgressReports instead of always? or maybe the number of steps should be small enough that we should show every time?
	// TODO: how do we get the gradient of multiple gaussians? sum the gradient of each gaussian. work out the formula for a single gaussian.
	// TODO: get the index to color stuff to show the dots of each type!
	// TODO: draw the starting points as gaussian blobs, with a sigma defined in the settings (needs to change with resolution)

	// Randomly init our starting points
	std::vector<float2> points(2);
	for (float2& p : points)
		p = float2{ distPos(rng), distPos(rng) };

	// Iterate
	char fileName[1024];
	for (int i = 0; i < c_2DNumSteps; ++i)
	{
		sprintf_s(fileName, "%s%u_%i.png", baseFileName, seed, i);
		DrawGaussians(gaussians, points, fileName);
	}

}

int main(int argc, char** argv)
{
	_mkdir("out");

	DoTest2D("out/2D_");
	
	return 0;
}

/*
TODO:
* move basic math stuff into it's own header
* 1d and higher comparison of GD (different learning rates) vs adam
* maybe sum of random gaussians?
* could also print out CSVs and graph error / convergence
* animated gif output? (stb, then make in gimp?)

Notes:
https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
https://machinelearningmastery.com/adam-optimization-from-scratch/



*/