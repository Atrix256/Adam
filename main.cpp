#define _CRT_SECURE_NO_WARNINGS // for stb

// Parameters
#define DETERMINISTIC() true
#define DETERMINISTIC_SEED() 927170769

static const int   c_2DNumSteps = 100;

static const float c_2DLearningRates[] = { 0.001f, 0.0001f, 0.01f };
static const int   c_2DPointsPerLearningRate = 20;

static const float c_2DAdamAlphas[] = { 0.01f, 0.001f, 0.1f };
static const int   c_2DNumAdamPoints = 20;

static const int   c_2DNumGaussians = 25;
static const float c_2DSigmaMin = 0.05f;
static const float c_2DSigmaMax = 0.2f;

static const int   c_2DImageSize = 256;
static const float c_2DPointRadius = 2.0f * float(c_2DImageSize) / 256.0f;
static const int   c_2DTopoColors = 16;

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
	float amplitude = 1.0f;
};

// Note: alpha needs to be tuned, but beta1, beta2, epsilon as usually fine as is
struct Adam
{
	Adam(float alpha = 0.01f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 0.0001f)
		: m_m(0.0f)
		, m_v(0.0f)
		, m_alpha(alpha)
		, m_beta1(beta1)
		, m_beta2(beta2)
		, m_epsilon(epsilon)
		, m_beta1Decayed(beta1)
		, m_beta2Decayed(beta2)
	{
	}

	float m_m = 0.0f;
	float m_v = 0.0f;
	float m_alpha = 0.0f;
	float m_beta1 = 0.0f;
	float m_beta2 = 0.0f;
	float m_epsilon = 0.0f;

	float m_beta1Decayed = 0.0f;
	float m_beta2Decayed = 0.0f;

	float AdjustDerivative(float derivative)
	{
		m_m = m_beta1 * m_m + (1.0f - m_beta1) * derivative;
		m_v = m_beta2 * m_v + (1.0f - m_beta2) * derivative * derivative;

		float mhat = m_m / (1.0f - m_beta1Decayed);
		float vhat = m_v / (1.0f - m_beta2Decayed);

		m_beta1Decayed *= m_beta1;
		m_beta2Decayed *= m_beta2;

		return m_alpha * mhat / (std::sqrt(vhat) + m_epsilon);
	}
};

struct AdamPoint
{
	Adam m_adamX;
	Adam m_adamY;
	float2 m_point;
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
		ret += gaussian.amplitude * Gaussian(gaussian.center - x, gaussian.sigma);
	return ret;
}

float2 FGradient(const std::vector<Gauss2D>& gaussians, const float2& x)
{
	float2 ret = float2{ 0.0f, 0.0f };
	for (const Gauss2D& gaussian : gaussians)
	{
		ret[0] += gaussian.amplitude * GaussianDerivative(gaussian.center[0] - x[0], gaussian.sigma[0]) * Gaussian(gaussian.center[1] - x[1], gaussian.sigma[1]);
		ret[1] += gaussian.amplitude * GaussianDerivative(gaussian.center[1] - x[1], gaussian.sigma[1]) * Gaussian(gaussian.center[0] - x[0], gaussian.sigma[0]);
	}
	return ret;
}

static void PlotCircle(std::vector<unsigned char>& image, int width, int height, int x, int y, float radius, unsigned char color[3])
{
	int sx = Clamp((int)std::floor(float(x) - radius), 0, width - 1);
	int sy = Clamp((int)std::floor(float(y) - radius), 0, width - 1);

	int ex = Clamp((int)std::ceil(float(x) + radius), 0, height - 1);
	int ey = Clamp((int)std::ceil(float(y) + radius), 0, height - 1);

	for (int iy = sy; iy <= ey; ++iy)
	{
		float dy = float(iy - y);

		for (int ix = sx; ix <= ex; ++ix)
		{
			float dx = float(ix - x);

			float distance = std::sqrt(dx * dx + dy * dy) - radius;

			float alpha = 1.0f - SmoothStep(-1.0f, 0.0f, distance);

			unsigned char* pixel = &image[(iy * width + ix) * 3];
			for (int i = 0; i < 3; ++i)
			{
				unsigned char oldColor = *pixel;
				unsigned char newColor = (unsigned char)Lerp(float(oldColor), float(color[i]), alpha);
				*pixel = newColor;
				pixel++;
			}
		}
	}
}

template <typename TDrawLambda>
void DrawGaussians(const std::vector<Gauss2D>& gaussians, const char* fileName, const TDrawLambda& DrawLambda)
{
	// Gather up the gaussian pixel values
	std::vector<float> pixelsF(c_2DImageSize * c_2DImageSize, 0.0f);
	float* pixelF = pixelsF.data();
	float minValue = FLT_MAX;
	float maxValue = -FLT_MAX;
	for (int i = 0; i < c_2DImageSize * c_2DImageSize; ++i)
	{
		int x = i % c_2DImageSize;
		int y = i / c_2DImageSize;

		float2 uv = float2{
			(float(x) + 0.5f) / float(c_2DImageSize),
			(float(y) + 0.5f) / float(c_2DImageSize)
		};

		*pixelF += F(gaussians, uv);

		minValue = std::min(minValue, *pixelF);
		maxValue = std::max(maxValue, *pixelF);

		pixelF++;
	}

	// normalize and convert to U8
	std::vector<unsigned char> pixels(c_2DImageSize * c_2DImageSize * 3, 0);
	for (size_t i = 0; i < c_2DImageSize * c_2DImageSize; ++i)
	{
		float pixelFNormalized = (pixelsF[i] - minValue) / (maxValue - minValue);
		float pixelFNormalizedTopo = pixelFNormalized;

		if (c_2DTopoColors > 0)
			pixelFNormalizedTopo = std::floor(pixelFNormalized * float(c_2DTopoColors)) / float(c_2DTopoColors);

		unsigned char U8 = (unsigned char)Clamp(255.0f * pixelFNormalized, 0.0f, 255.0f);
		unsigned char U8Topo = (unsigned char)Clamp(255.0f * pixelFNormalizedTopo, 0.0f, 255.0f);

		pixels[i * 3 + 0] = U8Topo;
		pixels[i * 3 + 1] = U8Topo;
		pixels[i * 3 + 2] = U8Topo;
	}

	DrawLambda(pixels, c_2DImageSize, c_2DImageSize, 3);

	stbi_write_png(fileName, c_2DImageSize, c_2DImageSize, 3, pixels.data(), 0);
}

void DoTest2D(const char* baseFileName)
{
	unsigned int seed = DETERMINISTIC_SEED();
	std::mt19937 rng = GetRNG(seed);

	printf("Seed = %u\n", seed);

	// TODO: temp!
	seed = 0;

	// Generate the randomized gaussians
	std::uniform_real_distribution<float> distPos(0.0f, 1.0f);
	std::uniform_real_distribution<float> distSigma(c_2DSigmaMin, c_2DSigmaMax);
	std::vector<Gauss2D> gaussians(c_2DNumGaussians);
	for (Gauss2D& gaussian : gaussians)
	{
		gaussian.center = float2{ distPos(rng), distPos(rng) };
		gaussian.sigma = float2{ distSigma(rng), distSigma(rng) };
	}

	// add another gaussian that is a bowl, to keep the points in
	{
		Gauss2D bowl;
		bowl.center = float2{ 0.5f, 0.5f };
		bowl.sigma = float2{ 0.5f, 0.5f };
		bowl.amplitude = -1.0f * std::sqrt(float(c_2DNumGaussians));
		gaussians.push_back(bowl);
	}

	// Randomly init starting points
	std::vector<std::vector<float2>> allPoints(_countof(c_2DLearningRates));
	for (std::vector<float2>& points : allPoints)
	{
		points.resize(c_2DPointsPerLearningRate);
		for (float2& p : points)
			p = float2{ distPos(rng), distPos(rng) };
	}

	std::vector<std::vector<AdamPoint>> allAdamPoints(_countof(c_2DAdamAlphas));
	for (size_t alphaIndex = 0; alphaIndex < _countof(c_2DAdamAlphas); ++alphaIndex)
	{
		std::vector<AdamPoint>& adamPoints = allAdamPoints[alphaIndex];
		float alpha = c_2DAdamAlphas[alphaIndex];

		adamPoints.resize(c_2DNumAdamPoints);
		for (AdamPoint& p : adamPoints)
		{
			p.m_point = float2{ distPos(rng), distPos(rng) };
			p.m_adamX.m_alpha = alpha;
			p.m_adamY.m_alpha = alpha;
		}
	}

	auto DrawPoints = [&](std::vector<unsigned char>& pixels, int width, int height, int components)
		{
			int pointIndex = 0;
			for (const std::vector<float2>& points : allPoints)
			{
				for (const float2& point : points)
				{
					int px = Clamp(int(point[0] * width), 0, width - 1);
					int py = Clamp(int(point[1] * height), 0, height - 1);

					RGBu8 color = IndexToColor(pointIndex);
					unsigned char color2[3] = { color.R, color.G, color.B };

					PlotCircle(pixels, width, height, px, py, c_2DPointRadius, color2);
				}
				pointIndex++;
			}
		}
	;
	
	auto DrawAdamPoints = [&](std::vector<unsigned char>& pixels, int width, int height, int components)
		{
			// draw the adam points
			int pointIndex = c_2DPointsPerLearningRate * _countof(c_2DLearningRates);
			for (std::vector<AdamPoint>& adamPoints : allAdamPoints)
			{
				for (const AdamPoint& point : adamPoints)
				{
					int px = Clamp(int(point.m_point[0] * width), 0, width - 1);
					int py = Clamp(int(point.m_point[1] * height), 0, height - 1);

					RGBu8 color = IndexToColor(pointIndex);
					unsigned char color2[3] = { color.R, color.G, color.B };

					PlotCircle(pixels, width, height, px, py, c_2DPointRadius, color2);
				}
				pointIndex++;
			}
		}
	;

	// Iterate
	char fileName[1024];
	for (int i = 0; i < c_2DNumSteps; ++i)
	{
		printf("\r%i/%i", i, c_2DNumSteps);

		// show where the points are
		sprintf_s(fileName, "out/%s_GD_%u_%i.png", baseFileName, seed, i);
		DrawGaussians(gaussians, fileName, DrawPoints);
		sprintf_s(fileName, "out/%s_Adam_%u_%i.png", baseFileName, seed, i);
		DrawGaussians(gaussians, fileName, DrawAdamPoints);

		// update the points
		for (size_t learningRateIndex = 0; learningRateIndex < _countof(c_2DLearningRates); ++learningRateIndex)
		{
			std::vector<float2>& points = allPoints[learningRateIndex];
			for (float2& p : points)
			{
				float2 grad = FGradient(gaussians, p);
				p = p + grad * c_2DLearningRates[learningRateIndex];
				p[0] = Clamp(p[0], 0.0f, 1.0f);
				p[1] = Clamp(p[1], 0.0f, 1.0f);
			}
		}

		// update the adam points
		for (std::vector<AdamPoint>& adamPoints : allAdamPoints)
		{
			for (AdamPoint& point : adamPoints)
			{
				float2 grad = FGradient(gaussians, point.m_point);

				float2 adjustedGrad;
				adjustedGrad[0] = point.m_adamX.AdjustDerivative(grad[0]);
				adjustedGrad[1] = point.m_adamY.AdjustDerivative(grad[1]);

				point.m_point = point.m_point + adjustedGrad;
				point.m_point[0] = Clamp(point.m_point[0], 0.0f, 1.0f);
				point.m_point[1] = Clamp(point.m_point[1], 0.0f, 1.0f);
			}
		}
	}
	printf("\r%i/%i\n", c_2DNumSteps, c_2DNumSteps);

	// show the final position
	sprintf_s(fileName, "out/%s_GD_%u_%i.png", baseFileName, seed, c_2DNumSteps);
	DrawGaussians(gaussians, fileName, DrawPoints);
	sprintf_s(fileName, "out/%s_Adam_%u_%i.png", baseFileName, seed, c_2DNumSteps);
	DrawGaussians(gaussians, fileName, DrawAdamPoints);
}

int main(int argc, char** argv)
{
	_mkdir("out");

	DoTest2D("2D");
	
	return 0;
}

/*
TODO:
* make the first gaussian be subtracted and make the whole area into a bowl? to keep the minimums in the texture region properly. unsure what to make the amplitude though.
* show points of gradient descent for a couple different learning rates, and for adam,
* make a CSV of movement length each step, to graph it?
* clean up the adam code. less storage, cleaner to do values in bulk etc.
 * AdjustDerivative() should be called something else. Maybe "apply gradient" or something, and actually adjust the position? idk.
* the combined point set is too confusing. too many point colors all at once.
* why are we adding gradient instead of subtracting it?
* could figure out how to do actual topo lines. like SDF lines.
* could also print out CSVs and graph error / convergence? if needed
 * maybe a line for each learning rate, and alpha. so like 6 or 8 lines or whatever.
* animated gif output? (stb, then make in gimp?)
 * could maybe make the GD and Adam images be side by side so the animated gif shows both in sync at the same time

Notes:
https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
https://machinelearningmastery.com/adam-optimization-from-scratch/

also: https://www.youtube.com/watch?v=JXQT_vxqwIs
 * talks about hyper parameters and how you need to tune alpha.

Blog:
* show 1d and 2d both? or maybe just 2d
* show that you are using fewer colors to show the topo map?
* could show a bunch of regular gradient descent points first, and show how faster learning rate ones don't settle down. before showing adam.
* say and show that formula for m and v is equivelent to lerping
* compare learning rates in gradient descenet.
* compare alpha values in adam.
* momentum is kind of a "local search" to see if anything nearby leads lower.

*/