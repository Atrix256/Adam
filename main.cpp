#define _CRT_SECURE_NO_WARNINGS // for stb

// Parameters
#define DETERMINISTIC() true
#define DETERMINISTIC_SEED() 927170769

static const int   c_2DNumSteps = 100;

static const float c_2DLearningRates[] = { 0.01f, 0.001f, 0.0001f };
static const int   c_2DPointsPerLearningRate = 20;

static const float c_2DAdamAlphas[] = { 0.1f, 0.01f, 0.001f };
static const int   c_2DNumAdamPoints = 20;

static const int   c_2DNumGaussians = 25;
static const float c_2DSigmaMin = 0.05f;
static const float c_2DSigmaMax = 0.2f;
static const int   c_2DNumGaussiansSampled = 25;

static const int   c_2DImageSize = 256;
static const int   c_2DImagePaddingH = 16;
static const int   c_2DImagePaddingV = 8;
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
	// Internal state
	float m = 0.0f;
	float v = 0.0f;

	// Internal state calculated for convenience
	// If you have a bunch of derivatives, you would probably want to store / calculate these once
	// for the entire gradient, instead of each derivative like this is doing.
	float beta1Decayed = 1.0f;
	float beta2Decayed = 1.0f;

	float GetAdjustedDerivative(float derivative, float alpha)
	{
		// Adam parameters
		static const float c_beta1 = 0.9f;
		static const float c_beta2 = 0.999f;
		static const float c_epsilon = 1e-8f;

		// exponential moving average of first and second moment
		m = c_beta1 * m + (1.0f - c_beta1) * derivative;
		v = c_beta2 * v + (1.0f - c_beta2) * derivative * derivative;

		// bias correction
		beta1Decayed *= c_beta1;
		beta2Decayed *= c_beta2;
		float mhat = m / (1.0f - beta1Decayed);
		float vhat = v / (1.0f - beta2Decayed);

		// Adam adjusted derivative
		return alpha * mhat / (std::sqrt(vhat) + c_epsilon);
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
		ret += gaussian.amplitude * Gaussian(x - gaussian.center, gaussian.sigma);
	return ret;
}

float2 FStochasticGradient(const std::vector<Gauss2D>& gaussians, const float2& x, std::mt19937& rng)
{
	std::vector<int> gaussiansShuffled(gaussians.size());
	for (int i = 0; i < (int)gaussiansShuffled.size(); ++i)
		gaussiansShuffled[i] = i;
	std::shuffle(gaussiansShuffled.begin(), gaussiansShuffled.end(), rng);
	gaussiansShuffled.resize(c_2DNumGaussiansSampled);

	float2 ret = float2{ 0.0f, 0.0f };
	for (int i : gaussiansShuffled)
	{
		const Gauss2D& gaussian = gaussians[gaussiansShuffled[i]];
		ret[0] += gaussian.amplitude * GaussianDerivative(x[0] - gaussian.center[0], gaussian.sigma[0]) * Gaussian(x[1] - gaussian.center[1], gaussian.sigma[1]);
		ret[1] += gaussian.amplitude * GaussianDerivative(x[1] - gaussian.center[1], gaussian.sigma[1]) * Gaussian(x[0] - gaussian.center[0], gaussian.sigma[0]);
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

template <typename TDrawLambdaLeft, typename TDrawLambdaRight, typename TDrawLambdaMiddle>
void DrawGaussians(const std::vector<Gauss2D>& gaussians, const char* fileName, const TDrawLambdaLeft& DrawLambdaLeft, const TDrawLambdaRight& DrawLambdaRight, const TDrawLambdaMiddle& DrawLambdaMiddle, int frameNum, int frameMax)
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
	std::vector<unsigned char> pixelsLeft(c_2DImageSize * c_2DImageSize * 3, 0);
	for (size_t i = 0; i < c_2DImageSize * c_2DImageSize; ++i)
	{
		float pixelFNormalized = (pixelsF[i] - minValue) / (maxValue - minValue);
		float pixelFNormalizedTopo = pixelFNormalized;

		if (c_2DTopoColors > 0)
			pixelFNormalizedTopo = std::floor(pixelFNormalized * float(c_2DTopoColors)) / float(c_2DTopoColors);

		unsigned char U8 = (unsigned char)Clamp(255.0f * pixelFNormalized, 0.0f, 255.0f);
		unsigned char U8Topo = (unsigned char)Clamp(255.0f * pixelFNormalizedTopo, 0.0f, 255.0f);

		pixelsLeft[i * 3 + 0] = U8Topo;
		pixelsLeft[i * 3 + 1] = U8Topo;
		pixelsLeft[i * 3 + 2] = U8Topo;
	}
	std::vector<unsigned char> pixelsRight = pixelsLeft;

	// Custom drawing on left
	DrawLambdaLeft(pixelsLeft, c_2DImageSize, c_2DImageSize, 3);

	// Custom drawing on right
	DrawLambdaRight(pixelsRight, c_2DImageSize, c_2DImageSize, 3);

	// stick the images together
	int combinedImageWidth = (c_2DImageSize * 2 + c_2DImagePaddingH);
	int combinedImageHeight = (c_2DImageSize + c_2DImagePaddingV);
	std::vector<unsigned char> pixels(combinedImageWidth * combinedImageHeight * 3, 64);
	for (size_t iy = 0; iy < c_2DImageSize; ++iy)
	{
		memcpy(&pixels[iy * combinedImageWidth * 3], &pixelsLeft[iy * c_2DImageSize * 3], c_2DImageSize * 3);
		memcpy(&pixels[(iy * combinedImageWidth + c_2DImageSize + c_2DImagePaddingH) * 3], &pixelsRight[iy * c_2DImageSize * 3], c_2DImageSize * 3);
	}

	// show the progress bar at the bottom
	{
		int doneX = int(float(combinedImageWidth) * float(frameNum) / float(frameMax));

		for (int iy = c_2DImageSize; iy < combinedImageHeight; ++iy)
		{
			unsigned char* pixel = &pixels[iy * combinedImageWidth * 3];

			for (int ix = 0; ix < combinedImageWidth; ++ix)
			{
				if (ix < doneX)
				{
					pixel[0] = 0;
					pixel[1] = 255;
					pixel[2] = 0;
				}
				else
				{
					pixel[0] = 0;
					pixel[1] = 64;
					pixel[2] = 0;
				}
				pixel += 3;
			}
		}
	}

	// Other Custom Drawing
	DrawLambdaMiddle(pixels, combinedImageWidth, combinedImageHeight, 3, c_2DImageSize, c_2DImagePaddingH, c_2DImageSize);

	// save it
	stbi_write_png(fileName, combinedImageWidth, combinedImageHeight, 3, pixels.data(), 0);
}

void DoTest2D()
{
	unsigned int seed = DETERMINISTIC_SEED();
	std::mt19937 rng = GetRNG(seed);
	printf("Seed = %u\n", seed);

	// Generate the randomized gaussians
	std::uniform_real_distribution<float> distPos(0.0f, 1.0f);
	std::uniform_real_distribution<float> distSigma(c_2DSigmaMin, c_2DSigmaMax);
	std::vector<Gauss2D> gaussians(c_2DNumGaussians);
	for (Gauss2D& gaussian : gaussians)
	{
		gaussian.center = float2{ distPos(rng), distPos(rng) };
		gaussian.sigma = float2{ distSigma(rng), distSigma(rng) };
	}

	// force the first gaussian to be a bowl to hold the points in better
	{
		gaussians[0].center = float2{0.5f, 0.5f};
		gaussians[0].sigma = float2{ 0.5f, 0.5f };
		gaussians[0].amplitude = -1.0f * std::sqrt(float(c_2DNumGaussians));
	}

	// Randomly init starting points
	std::vector<float> allPointsAvgHeights(_countof(c_2DLearningRates), 0.0f);
	std::vector<std::vector<float2>> allPoints(_countof(c_2DLearningRates));
	for (std::vector<float2>& points : allPoints)
	{
		points.resize(c_2DPointsPerLearningRate);
		for (float2& p : points)
			p = float2{ distPos(rng), distPos(rng) };
	}

	std::vector<float> allAdamPointsAvgHeights(_countof(c_2DAdamAlphas), 0.0f);
	std::vector<std::vector<AdamPoint>> allAdamPoints(_countof(c_2DAdamAlphas));
	for (size_t alphaIndex = 0; alphaIndex < _countof(c_2DAdamAlphas); ++alphaIndex)
	{
		std::vector<AdamPoint>& adamPoints = allAdamPoints[alphaIndex];

		adamPoints.resize(c_2DNumAdamPoints);
		for (AdamPoint& p : adamPoints)
			p.m_point = float2{ distPos(rng), distPos(rng) };
	}

	auto DrawPoints = [&](std::vector<unsigned char>& pixels, int width, int height, int components)
		{
			// draw the points
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
			int pointIndex = _countof(c_2DLearningRates);
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

	auto DrawAvgs = [&](std::vector<unsigned char>& pixels, int width, int height, int components, int paddingBeginX, int paddingWidth, int heightWithoutPadding)
		{
			// get the min and max value of both GD and adam points
			float themin = FLT_MAX;
			float themax = -FLT_MAX;
			for (float f : allPointsAvgHeights)
			{
				themin = std::min(themin, f);
				themax = std::max(themax, f);
			}
			for (float f : allAdamPointsAvgHeights)
			{
				themin = std::min(themin, f);
				themax = std::max(themax, f);
			}

			// pad the min and the max to make the lines more readable
			float minmaxdiff = themax - themin;
			themax += minmaxdiff * 0.05f;
			themin -= minmaxdiff * 0.05f;

			// Draw lines in the padding for the normalized avg height of each point group
			int pointIndex = 0;
			for (float f : allPointsAvgHeights)
			{
				RGBu8 color = IndexToColor(pointIndex);
				unsigned char color2[3] = { color.R, color.G, color.B };

				int iy = heightWithoutPadding - int(float(heightWithoutPadding) * (f - themin) / (themax - themin));

				unsigned char* pixel = &pixels[(iy * width + paddingBeginX) * 3];
				for (int ix = 0; ix < paddingWidth; ++ix)
				{
					memcpy(pixel, color2, 3);
					pixel += 3;
				}

				pointIndex++;
			}

			for (float f : allAdamPointsAvgHeights)
			{
				RGBu8 color = IndexToColor(pointIndex);
				unsigned char color2[3] = { color.R, color.G, color.B };

				int iy = heightWithoutPadding - int(float(heightWithoutPadding) * (f - themin) / (themax - themin));

				unsigned char* pixel = &pixels[(iy * width + paddingBeginX) * 3];
				for (int ix = 0; ix < paddingWidth; ++ix)
				{
					memcpy(pixel, color2, 3);
					pixel += 3;
				}

				pointIndex++;
			}
		}
	;

	// Open the csv
	char fileName[1024];
	FILE* csvFile = nullptr;
	sprintf_s(fileName, "out/2D_%u.csv", seed);
	fopen_s(&csvFile, fileName, "wb");
	fprintf(csvFile, "\"Step\"");
	for (size_t learningRateIndex = 0; learningRateIndex < _countof(c_2DLearningRates); ++learningRateIndex)
		fprintf(csvFile, ",\"GD LR %f\"", c_2DLearningRates[learningRateIndex]);
	for (size_t learningRateIndex = 0; learningRateIndex < _countof(c_2DAdamAlphas); ++learningRateIndex)
		fprintf(csvFile, ",\"Adam Alpha %f\"", c_2DAdamAlphas[learningRateIndex]);
	fprintf(csvFile, "\n");

	// write the initial average heights
	{
		fprintf(csvFile, "\"%i\"", 0);

		for (size_t learningRateIndex = 0; learningRateIndex < _countof(c_2DLearningRates); ++learningRateIndex)
		{
			float& avgHeight = allPointsAvgHeights[learningRateIndex];
			avgHeight = 0.0f;
			std::vector<float2>& points = allPoints[learningRateIndex];
			for (int pointIndex = 0; pointIndex < points.size(); ++pointIndex)
			{
				float2& p = points[pointIndex];
				avgHeight = Lerp(avgHeight, F(gaussians, p), 1.0f / float(pointIndex + 1));
			}
			fprintf(csvFile, ",\"%f\"", avgHeight);
		}

		for (size_t alphaIndex = 0; alphaIndex < _countof(c_2DAdamAlphas); ++alphaIndex)
		{
			std::vector<AdamPoint>& adamPoints = allAdamPoints[alphaIndex];

			float& avgHeight = allAdamPointsAvgHeights[alphaIndex];
			avgHeight = 0.0f;
			for (int pointIndex = 0; pointIndex < adamPoints.size(); ++pointIndex)
			{
				float2& p = adamPoints[pointIndex].m_point;
				avgHeight = Lerp(avgHeight, F(gaussians, p), 1.0f / float(pointIndex + 1));
			}
			fprintf(csvFile, ",\"%f\"", avgHeight);
		}
		fprintf(csvFile, "\n");
	}

	// Iterate
	for (int i = 0; i < c_2DNumSteps; ++i)
	{
		printf("\rStep: %i/%i", i, c_2DNumSteps);

		fprintf(csvFile, "\"%i\"", i + 1);

		// show where the points are
		sprintf_s(fileName, "out/2D_%u_%i.png", seed, i);
		DrawGaussians(gaussians, fileName, DrawPoints, DrawAdamPoints, DrawAvgs, i, c_2DNumSteps);

		// update the points
		for (size_t learningRateIndex = 0; learningRateIndex < _countof(c_2DLearningRates); ++learningRateIndex)
		{
			float& avgHeight = allPointsAvgHeights[learningRateIndex];
			avgHeight = 0.0f;

			std::vector<float2>& points = allPoints[learningRateIndex];
			for (int pointIndex = 0; pointIndex < points.size(); ++pointIndex)
			{
				float2& p = points[pointIndex];

				float2 grad = FStochasticGradient(gaussians, p, rng);
				p = p - grad * c_2DLearningRates[learningRateIndex];

				p[0] = Clamp(p[0], 0.0f, 1.0f);
				p[1] = Clamp(p[1], 0.0f, 1.0f);

				avgHeight = Lerp(avgHeight, F(gaussians, p), 1.0f / float(pointIndex + 1));
			}

			fprintf(csvFile, ",\"%f\"", avgHeight);
		}

		// update the adam points
		for (size_t alphaIndex = 0; alphaIndex < _countof(c_2DAdamAlphas); ++alphaIndex)
		{
			float alpha = c_2DAdamAlphas[alphaIndex];

			std::vector<AdamPoint>& adamPoints = allAdamPoints[alphaIndex];

			float& avgHeight = allAdamPointsAvgHeights[alphaIndex];
			avgHeight = 0.0f;

			for (int pointIndex = 0; pointIndex < adamPoints.size(); ++pointIndex)
			{
				AdamPoint& point = adamPoints[pointIndex];

				float2 grad = FStochasticGradient(gaussians, point.m_point, rng);

				float2 adjustedGrad;
				adjustedGrad[0] = point.m_adamX.GetAdjustedDerivative(grad[0], alpha);
				adjustedGrad[1] = point.m_adamY.GetAdjustedDerivative(grad[1], alpha);

				point.m_point = point.m_point - adjustedGrad;
				point.m_point[0] = Clamp(point.m_point[0], 0.0f, 1.0f);
				point.m_point[1] = Clamp(point.m_point[1], 0.0f, 1.0f);

				avgHeight = Lerp(avgHeight, F(gaussians, point.m_point), 1.0f / float(pointIndex + 1));
			}

			fprintf(csvFile, ",\"%f\"", avgHeight);
		}

		fprintf(csvFile, "\n");
	}
	printf("\rStep: %i/%i\n", c_2DNumSteps, c_2DNumSteps);

	// close the CSV
	fclose(csvFile);

	// show the final position
	sprintf_s(fileName, "out/2D_%u_%i.png", seed, c_2DNumSteps);
	DrawGaussians(gaussians, fileName, DrawPoints, DrawAdamPoints, DrawAvgs, c_2DNumSteps, c_2DNumSteps);
}

int main(int argc, char** argv)
{
	_mkdir("out");

	DoTest2D();

	/*
	for (int i = 0; i < 6; ++i)
	{
		RGBu8 color = IndexToColor(i);
		printf("[%i] : (%f, %f, %f)\n", i, color.R / 255.0f, color.G / 255.0f, color.B / 255.0f);
	}
	*/
	
	return 0;
}
