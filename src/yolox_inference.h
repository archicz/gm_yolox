#ifndef YOLOX_INFERENCE_H
#define YOLOX_INFERENCE_H

#include <cstdint>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

#define YOLOX_INFERENCE_USE_DML
#define YOLOX_USE_STB_RESIZE2

class yolox_inference
{
public:
	struct object_proposed
	{
		float x;
		float y;
		float w;
		float h;
		float prob;
		size_t label;
	public:
		float area();
		float area_intersect(object_proposed& other);
	};
	
	class model_info
	{
	public:
		struct grid_and_stride
		{
			int32_t grid0;
			int32_t grid1;
			int32_t stride;
		};
	private:
		std::wstring modelPath;
		uint32_t modelWidth;
		uint32_t modelHeight;
		std::vector<grid_and_stride> modelStrides;
		std::vector<float> modelMean;
		std::vector<float> modelNorm;
	public:
		static constexpr uint8_t modelChannels = 3;
		static constexpr uint32_t modelStride[modelChannels] = { 8, 16, 32 };
	public:
		model_info(std::wstring path, uint32_t width, uint32_t height);
	public:
		std::wstring get_path();
		uint32_t get_width();
		uint32_t get_height();
		std::vector<grid_and_stride>& get_strides();
	public:
		std::vector<float>& get_mean();
		std::vector<float>& get_norm();
	public:
		void set_mean(std::vector<float> mean);
		void set_norm(std::vector<float> norm);
	};
protected:
	bool isReady;
	
	uint32_t inputWidth;
	uint32_t inputHeight;
	float inputScaleWidth;
	float inputScaleHeight;
	float* inputData;

	model_info mdlInfo;
	char* mdlInputName;
	char* mdlOutputName;

	uint32_t inputWidthOverride;
	uint32_t inputHeightOverride;
	bool scaleOverride;

	uint8_t deviceId;
	OrtSessionOptions* ortSessionOptions;
	OrtSession* ortSession;
	Ort::AllocatorWithDefaultOptions ortAllocator;

	std::vector<object_proposed> objects;
	float probThreshold;
	float nmsThreshold;
public:
	yolox_inference(uint8_t deviceId, std::wstring modelPath, uint32_t modelWidth, uint32_t modelHeight);
	virtual ~yolox_inference();
protected:
	void process_proposals(float* proposals, size_t numProposals);
	void nms_objects();
	void scale_objects();
private:
	virtual void append_executor();
public:
	bool create_session();
	bool inference_float(float* pixelData, uint32_t width, uint32_t height);
	bool inference_rgbx8888(uint8_t* pixelData, uint32_t width, uint32_t height);
	std::vector<object_proposed>& get_objects();
public:
	float get_probability_threshold();
	float get_nms_threshold();

	uint32_t get_input_width();
	uint32_t get_input_height();
public:
	void set_probability_threshold(float threshold);
	void set_nms_threshold(float threshold);

	void set_scale_override(bool enabled, uint32_t width, uint32_t height);
public:
	model_info& get_model_info();
};

#ifdef YOLOX_INFERENCE_USE_DML
#include <dml_provider_factory.h>
class yolox_inference_dml : public yolox_inference
{
public:
	yolox_inference_dml(uint8_t deviceId, std::wstring modelPath, uint32_t modelWidth, uint32_t modelHeight);
public:
	virtual void append_executor() override;
};
#endif

#endif