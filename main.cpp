#define _CRT_SECURE_NO_WARNINGS // for stb

// Parameters
#define DETERMINISTIC() false

static const int   c_2DImageSize = 256;
static const float c_2DPointSigma = 2.0f;
static const int   c_2DNumGaussians = 25;
static const float c_2DSigmaMin = 0.05f;
static const float c_2DSigmaMax = 0.2f;
static const int   c_2DNumSteps = 25;
static const float c_2DLearningRate = 1.0f;
static const int   c_2DTopoColors = 8;

#include <stdio.h>
#include <array>
#include <random>
#include <vector>
#include <direct.h>

#include "maths.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

struct Gauss2D
{
	float2 center;
	float2 sigma;
};

std::mt19937 GetRNG(unsigned int& seed)
{
#if !DETERMINISTIC()
	std::random_device rd;
	seed = rd();
#endif
	return std::mt19937(seed);
}

float F(const std::vector<Gauss2D>& gaussians, const float2& x)
{
	float ret = 0.0f;
	for (const Gauss2D& gaussian : gaussians)
		ret += Gaussian(gaussian.center - x, gaussian.sigma);
	return ret;
}

float2 FGradient(const std::vector<Gauss2D>& gaussians, const float2& x)
{
	float2 ret = float2{ 0.0f, 0.0f };
	for (const Gauss2D& gaussian : gaussians)
	{
		ret[0] += GaussianDerivative(gaussian.center[0] - x[0], gaussian.sigma[0]);
		ret[1] += GaussianDerivative(gaussian.center[1] - x[1], gaussian.sigma[1]);
	}
	return ret;
}

static void PlotGaussian(std::vector<unsigned char>& image, int width, int height, int x, int y, float sigma, unsigned char color[3])
{
	int kernelRadius = int(std::sqrt(-2.0f * sigma * sigma * std::log(0.005f)));

	int sx = Clamp(x - kernelRadius, 0, width - 1);
	int ex = Clamp(x + kernelRadius, 0, height - 1);
	int sy = Clamp(y - kernelRadius, 0, width - 1);
	int ey = Clamp(y + kernelRadius, 0, height - 1);

	for (int iy = sy; iy <= ey; ++iy)
	{
		unsigned char* pixel = &image[(iy * width + sx) * 3];

		int ky = std::abs(iy - y);
		float kernelY = std::exp(-float(ky * ky) / (2.0f * sigma * sigma));

		for (int ix = sx; ix <= ex; ++ix)
		{
			int kx = std::abs(ix - x);
			float kernelX = std::exp(-float(kx * kx) / (2.0f * sigma * sigma));

			float kernel = kernelX * kernelY;

			for (int i = 0; i < 3; ++i)
			{
				unsigned char oldColor = *pixel;
				unsigned char newColor = (unsigned char)Lerp(float(oldColor), float(color[i]), kernel);
				*pixel = newColor;
				pixel++;
			}
		}
	}
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

		*pixelF += F(gaussians, uv);

		maxValue = std::max(maxValue, *pixelF);

		pixelF++;
	}

	// normalize and convert to U8
	std::vector<unsigned char> pixels(c_2DImageSize * c_2DImageSize * 3, 0);
	for (size_t i = 0; i < c_2DImageSize * c_2DImageSize; ++i)
	{
		float pixelFNormalized = pixelsF[i] / maxValue;

		if (c_2DTopoColors > 0)
			pixelFNormalized = std::floor(pixelFNormalized * float(c_2DTopoColors)) / float(c_2DTopoColors);

		unsigned char U8 = (unsigned char)Clamp(255.0f * pixelFNormalized, 0.0f, 255.0f);
		pixels[i * 3 + 0] = U8;
		pixels[i * 3 + 1] = U8;
		pixels[i * 3 + 2] = U8;
	}

	// draw the points
	int pointIndex = -1;
	for (const float2& point : points)
	{
		pointIndex++;
		int px = Clamp(int(point[0] * c_2DImageSize), 0, c_2DImageSize - 1);
		int py = Clamp(int(point[1] * c_2DImageSize), 0, c_2DImageSize - 1);

		RGBu8 color = IndexToColor(pointIndex);
		unsigned char color2[3] = { color.R, color.G, color.B };

		PlotGaussian(pixels, c_2DImageSize, c_2DImageSize, px, py, c_2DPointSigma, color2);
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

	// Randomly init our starting points
	std::vector<float2> points(20);
	for (float2& p : points)
		p = float2{ distPos(rng), distPos(rng) };

	// Iterate
	char fileName[1024];
	for (int i = 0; i < c_2DNumSteps; ++i)
	{
		sprintf_s(fileName, "%s%u_%i.png", baseFileName, seed, i);
		DrawGaussians(gaussians, points, fileName);

		for (float2& p : points)
		{
			float2 grad = FGradient(gaussians, p);
			p = p + grad * c_2DLearningRate;
			p[0] = Clamp(p[0], 0.0f, 1.0f);
			p[1] = Clamp(p[1], 0.0f, 1.0f);
		}
	}

	sprintf_s(fileName, "%s%u_%i.png", baseFileName, seed, c_2DNumSteps);
	DrawGaussians(gaussians, points, fileName);
}

int main(int argc, char** argv)
{
	_mkdir("out");

	DoTest2D("out/2D_");
	
	return 0;
}

/*
TODO:
* I don't think the gradient is correct. points aren't settling into local minimas in the gaussians
 * the gradient seems to have a reversed sign i think? it's going up hill when i subtract it.
? should we have multiple points with the same learning rate (showing a flock of rolling balls? or just a single one?)
* do 1d examples first?
* maybe sum of random gaussians?
* could also print out CSVs and graph error / convergence
* animated gif output? (stb, then make in gimp?)

Notes:
https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
https://machinelearningmastery.com/adam-optimization-from-scratch/

Blog:
* show 1d and 2d both
* show that you are using fewer colors to show the topo map?


*/