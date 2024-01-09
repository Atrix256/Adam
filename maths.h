#pragma once

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

template <size_t N>
Vec<N> operator + (const Vec<N>& A, const Vec<N>& B)
{
	Vec<N> ret;
	for (size_t i = 0; i < N; ++i)
		ret[i] = A[i] + B[i];
	return ret;
}

template <size_t N>
Vec<N> operator * (const Vec<N>& A, float B)
{
	Vec<N> ret;
	for (size_t i = 0; i < N; ++i)
		ret[i] = A[i] * B;
	return ret;
}

template <size_t N>
float Length(const Vec<N>& V)
{
	float ret = 0.0f;
	for (float f : V)
		ret += f * f;
	return std::sqrt(ret);
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

inline float Lerp(float A, float B, float t)
{
	return A * (1.0f - t) + B * t;
}

inline float SmoothStep(float edge0, float edge1, float x)
{
	x = Clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
	return x * x * (3.0f - 2.0f * x);
}

inline float Gaussian(float x, float sigma)
{
	return std::exp(-(x * x) / (2.0f * sigma * sigma));
}

template <size_t N>
float Gaussian(const Vec<N>& x, const Vec<N>& sigma)
{
	float ret = 1.0f;
	for (size_t i = 0; i < N; ++i)
		ret *= Gaussian(x[i], sigma[i]);
	return ret;
}

inline float GaussianDerivative(float x, float sigma)
{
	float denom = (2.0f * sigma * sigma);
	return -2.0f / denom * x * Gaussian(x, sigma);
}

struct RGBf
{
	float R;
	float G;
	float B;
};

// https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
inline RGBf HSVToRGB(float h, float s, float v)
{
	float h_i = floorf(h * 6.0f);
	float f = h * 6.0f - h_i;
	float p = v * (1.0f - s);
	float q = v * (1.0f - f * s);
	float t = v * (1.0f - (1.0f - f) * s);
	switch (int(h_i))
	{
		case 0: return RGBf{ v, t, p };
		case 1: return RGBf{ q, v, p };
		case 2: return RGBf{ p, v, t };
		case 3: return RGBf{ p, q, v };
		case 4: return RGBf{ t, p, v };
		case 5: return RGBf{ v, p, q };
		default: return RGBf{ 1.0f, 0.0f, 1.0f }; // doesn't happen
	}
}

// SRGB adapted from https://github.com/TheRealMJP/BakingLab
inline float SRGBToLinear(float color)
{
	float x = color / 12.92f;
	float y = std::pow((color + 0.055f) / 1.055f, 2.4f);
	return color <= 0.04045f ? x : y;
}

inline float LinearTosRGB(float color)
{
	float x = color * 12.92f;
	float y = std::pow(color, 1.0f / 2.4f) * 1.055f - 0.055f;
	return color < 0.0031308f ? x : y;
}

struct RGBu8
{
	unsigned char R;
	unsigned char G;
	unsigned char B;
};

// This uses the golden ratio to make N hues that are maximally distant
// from each other for any N colors desired. Need to use indices [0,N)
// for this to work though. Just does a 1D low discrepancy sequence
// for hue, and has constant s and v values.
inline RGBu8 IndexToColor(int index, float s = 1.0f, float v = 1.0f)
{
	static const float c_goldenRatioConjugate = 0.61803398875f;
	float h = std::fmodf(float(index) * c_goldenRatioConjugate, 1.0f);
	RGBf retf = HSVToRGB(h, s, v);
	RGBu8 ret;
	ret.R = (unsigned char)Clamp(retf.R * 255.0f, 0.0f, 255.0f);
	ret.G = (unsigned char)Clamp(retf.G * 255.0f, 0.0f, 255.0f);
	ret.B = (unsigned char)Clamp(retf.B * 255.0f, 0.0f, 255.0f);
	return ret;
}
