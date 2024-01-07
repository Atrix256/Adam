#define _CRT_SECURE_NO_WARNINGS // for stb

// Parameters
#define DETERMINISTIC() true
#define DETERMINISTIC_SEED() 2629819142  // Nice geometry, and a pink ball doesn't settle in correctly


static const int   c_2DNumSteps = 100;
static const float c_2DLearningRates[] = { 0.005f, 0.01f, 0.02f, 0.2f, 0.005f, 0.01f, 0.02f, 0.2f, 0.005f, 0.01f, 0.02f, 0.2f, 0.005f, 0.01f, 0.02f, 0.2f };
//static const int
static const float c_2DLearningRateMultiplier = 1.0f / 10.0f;
static const int   c_2DNumAdamPoints = 10;

static const int   c_2DNumGaussians = 25;
static const float c_2DSigmaMin = 0.05f;
static const float c_2DSigmaMax = 0.2f;

static const int   c_2DImageSize = 256;
static const float c_2DPointSigma = 2.0f;
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
};

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
		ret += Gaussian(gaussian.center - x, gaussian.sigma);
	return ret;
}

float2 FGradient(const std::vector<Gauss2D>& gaussians, const float2& x)
{
	float2 ret = float2{ 0.0f, 0.0f };
	for (const Gauss2D& gaussian : gaussians)
	{
		ret[0] += GaussianDerivative(gaussian.center[0] - x[0], gaussian.sigma[0]) * Gaussian(gaussian.center[1] - x[1], gaussian.sigma[1]);
		ret[1] += GaussianDerivative(gaussian.center[1] - x[1], gaussian.sigma[1]) * Gaussian(gaussian.center[0] - x[0], gaussian.sigma[0]);
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

void DrawGaussians(const std::vector<Gauss2D>& gaussians, const std::vector<float2>& points, const std::vector<AdamPoint>& adamPoints, const char* fileName)
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
		float pixelFNormalizedTopo = pixelFNormalized;

		if (c_2DTopoColors > 0)
			pixelFNormalizedTopo = std::floor(pixelFNormalized * float(c_2DTopoColors)) / float(c_2DTopoColors);

		unsigned char U8 = (unsigned char)Clamp(255.0f * pixelFNormalized, 0.0f, 255.0f);
		unsigned char U8Topo = (unsigned char)Clamp(255.0f * pixelFNormalizedTopo, 0.0f, 255.0f);

		pixels[i * 3 + 0] = U8Topo;
		pixels[i * 3 + 1] = U8Topo;
		pixels[i * 3 + 2] = U8Topo;
	}

	// draw the points
	int pointIndex = -1;
	for (const float2& point : points)
	{
		pointIndex++;
		int px = Clamp(int(point[0] * c_2DImageSize), 0, c_2DImageSize - 1);
		int py = Clamp(int(point[1] * c_2DImageSize), 0, c_2DImageSize - 1);

		RGBu8 color = IndexToColor(0);
		unsigned char color2[3] = { color.R, color.G, color.B };

		PlotGaussian(pixels, c_2DImageSize, c_2DImageSize, px, py, c_2DPointSigma, color2);
	}

	// draw the adam points
	for (const AdamPoint& point : adamPoints)
	{
		pointIndex++;
		int px = Clamp(int(point.m_point[0] * c_2DImageSize), 0, c_2DImageSize - 1);
		int py = Clamp(int(point.m_point[1] * c_2DImageSize), 0, c_2DImageSize - 1);

		RGBu8 color = IndexToColor(1);
		unsigned char color2[3] = { color.R, color.G, color.B };

		PlotGaussian(pixels, c_2DImageSize, c_2DImageSize, px, py, c_2DPointSigma, color2);
	}

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

		// TODO: temp!

		//gaussian.center = float2{ 0.0f, 0.0f };
		//gaussian.sigma = float2{ 0.25f, 100.0f };
		//gaussian.sigma = float2{ 100.0f, 0.25f };

	}

	// TODO: temp!
	/*
	
	gaussians[0].center = float2{0.0f, 0.0f};
	gaussians[0].sigma = float2{ 100.0f, 0.25f };

	gaussians[1].center = float2{ 0.0f, 1.0f };
	gaussians[1].sigma = float2{ 100.0f, 0.25f };
	*/

	// TODO: show points of gradient descent for a couple different learning rates, and for adam,
	// TODO: make a CSV of movement length each step, to graph it?
	// TODO: report only c_2DNumProgressReports instead of always? or maybe the number of steps should be small enough that we should show every time?
	// TODO: how do we get the gradient of multiple gaussians? sum the gradient of each gaussian. work out the formula for a single gaussian.
	// TODO: maybe have the regular GD points be one color, and the adam points be another color

	// Randomly init our starting points
	std::vector<float2> points(_countof(c_2DLearningRates));
	for (float2& p : points)
		p = float2{ distPos(rng), distPos(rng) };

	std::vector<AdamPoint> adamPoints(c_2DNumAdamPoints);
	for (AdamPoint& p : adamPoints)
		p.m_point = float2{ distPos(rng), distPos(rng) };

	// Iterate
	char fileName[1024];
	for (int i = 0; i < c_2DNumSteps; ++i)
	{
		// show where the points are
		sprintf_s(fileName, "%s%u_%i.png", baseFileName, seed, i);
		DrawGaussians(gaussians, points, adamPoints, fileName);

		// update the points
		for (size_t learningRateIndex = 0; learningRateIndex < _countof(c_2DLearningRates); ++learningRateIndex)
		{
			float2& p = points[learningRateIndex];
			float2 grad = FGradient(gaussians, p);
			p = p + grad * c_2DLearningRates[learningRateIndex] * c_2DLearningRateMultiplier;
			p[0] = Clamp(p[0], 0.0f, 1.0f);
			p[1] = Clamp(p[1], 0.0f, 1.0f);
		}

		// update the adam points
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

	// show the final position
	sprintf_s(fileName, "%s%u_%i.png", baseFileName, seed, c_2DNumSteps);
	DrawGaussians(gaussians, points, adamPoints, fileName);
}

int main(int argc, char** argv)
{
	_mkdir("out");

	DoTest2D("out/2D_");
	
	return 0;
}

/*
TODO:
* make DrawGaussians take a vector of vector of points.
 * all [0] points colored the same, all [1] points colored the same, ... all [n] points colored the same.
 * have a points set for each learning rate, and another for adam points
* why are we adding gradient instead of subtracting it?
* could figure out how to do actual topo lines. like SDF lines.
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
* show 1d and 2d both? or maybe just 2d
* show that you are using fewer colors to show the topo map?
* could show a bunch of regular gradient descent points first, and show how faster learning rate ones don't settle down. before showing adam.
* say and show that formula for m and v is equivelent to lerping


*/