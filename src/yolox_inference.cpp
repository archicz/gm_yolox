#include "yolox_inference.h"
#include <algorithm>

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#ifdef YOLOX_USE_STB_RESIZE2
#include "stb_image_resize2.h"
#endif

static const OrtApi* ortApi = nullptr;
static OrtEnv* ortEnv = nullptr;

yolox_inference::model_info::model_info(std::wstring path, uint32_t width, uint32_t height) :
	modelPath(path), modelWidth(width), modelHeight(height)
{
	for (int32_t stride : modelStride)
	{
		int32_t gridW = modelWidth / stride;
		int32_t gridH = modelHeight / stride;

		for (int32_t g1 = 0; g1 < gridH; g1++)
		{
			for (int32_t g0 = 0; g0 < gridW; g0++)
			{
				modelStrides.push_back(grid_and_stride{ g0, g1, stride });
			}
		}
	}
}



std::wstring yolox_inference::model_info::get_path()
{
	return modelPath;
}

uint32_t yolox_inference::model_info::get_width()
{
	return modelWidth;
}

uint32_t yolox_inference::model_info::get_height()
{
	return modelHeight;
}

std::vector<yolox_inference::model_info::grid_and_stride>& yolox_inference::model_info::get_strides()
{
	return modelStrides;
}

std::vector<float>& yolox_inference::model_info::get_mean()
{
	return modelMean;
}

std::vector<float>& yolox_inference::model_info::get_norm()
{
	return modelNorm;
}

void yolox_inference::model_info::set_mean(std::vector<float> mean)
{
	modelMean.swap(mean);
}

void yolox_inference::model_info::set_norm(std::vector<float> norm)
{
	modelNorm.swap(norm);
}



float yolox_inference::object_proposed::area()
{
	return w * h;
}

float yolox_inference::object_proposed::area_intersect(object_proposed& other)
{
	float x1 = std::max(x, other.x);
	float y1 = std::max(y, other.y);
	float x2 = std::min(x + w, other.x + other.w);
	float y2 = std::min(y + h, other.y + other.h);

	float intW = std::max(0.f, x2 - x1);
	float intH = std::max(0.f, y2 - y1);

	return intW * intH;
}



yolox_inference::yolox_inference(uint8_t devId, std::wstring modelPath, uint32_t modelWidth, uint32_t modelHeight) :
	isReady(false),
	mdlInfo(modelPath, modelWidth, modelHeight),
	mdlInputName(nullptr), mdlOutputName(nullptr),
	inputWidth(0), inputHeight(0), inputScaleWidth(0.f), inputScaleHeight(0.f),
	inputData(nullptr),
	deviceId(devId),
	ortSession(nullptr), ortSessionOptions(nullptr),
	probThreshold(0.5f), nmsThreshold(0.7f),
	inputWidthOverride(0), inputHeightOverride(0), scaleOverride(false)
{
	inputData = new float[modelWidth * modelHeight * model_info::modelChannels];

	if (!ortApi)
	{
		ortApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	}

	if (!ortEnv)
	{
		ortApi->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "yolox-inference", &ortEnv);
		ortApi->DisableTelemetryEvents(ortEnv);
	}

	ortApi->CreateSessionOptions(&ortSessionOptions);
	append_executor();
}

yolox_inference::~yolox_inference()
{
	if (ortSession)
	{
		ortApi->ReleaseSession(ortSession);
		ortSession = nullptr;
	}

	if (ortSessionOptions)
	{
		ortApi->ReleaseSessionOptions(ortSessionOptions);
		ortSessionOptions = nullptr;
	}

	if (mdlInputName)
	{
		ortAllocator.Free(mdlInputName);
		mdlInputName = nullptr;
	}

	if (mdlOutputName)
	{
		ortAllocator.Free(mdlOutputName);
		mdlOutputName = nullptr;
	}

	if (inputData)
	{
		delete[] inputData;
	}
}

void yolox_inference::process_proposals(float* proposals, size_t numProposals)
{
	objects.clear();
	auto& strides = mdlInfo.get_strides();

	for (size_t anchorIdx = 0; anchorIdx < strides.size(); anchorIdx++)
	{
		int32_t grid0 = strides[anchorIdx].grid0;
		int32_t grid1 = strides[anchorIdx].grid1;
		int32_t stride = strides[anchorIdx].stride;
		size_t startIdx = anchorIdx * numProposals;

		float w = exp(proposals[startIdx + 2]) * stride;
		float h = exp(proposals[startIdx + 3]) * stride;
		float x0 = ((proposals[startIdx + 0] + grid0) * stride) - w * 0.5f;
		float y0 = ((proposals[startIdx + 1] + grid1) * stride) - h * 0.5f;
		float objectness = proposals[startIdx + 4];

		object_proposed obj;
		obj.x = x0;
		obj.y = y0;
		obj.w = w;
		obj.h = h;
		obj.label = 0;
		obj.prob = 0;

		for (size_t classIdx = 0; classIdx < (numProposals - 5); classIdx++)
		{
			float boxClsScore = proposals[startIdx + 5 + classIdx];
			float boxProbability = objectness * boxClsScore;

			if (boxProbability > obj.prob)
			{
				obj.label = classIdx;
				obj.prob = boxProbability;
			}
		}

		if (obj.prob > probThreshold)
		{
			objects.push_back(obj);
		}
	}

	std::sort(objects.begin(), objects.end(), [](object_proposed& a, object_proposed& b) -> bool
	{
		return a.prob > b.prob;
	});
}

void yolox_inference::nms_objects()
{
	std::vector<object_proposed> nmsObjects;

	for (object_proposed& objA : objects)
	{
		bool keep = true;

		for (object_proposed& objB : nmsObjects)
		{
			float inter_area = objA.area_intersect(objB);
			float union_area = objA.area() + objB.area() - inter_area;

			if (inter_area / union_area > nmsThreshold)
			{
				keep = false;
			}
		}

		if (keep)
		{
			nmsObjects.push_back(objA);
		}
	}

	objects.swap(nmsObjects);
}

void yolox_inference::scale_objects()
{
	float finalScaleWidth = 0.f;
	float finalScaleHeight = 0.f;

	if (scaleOverride)
	{
		uint32_t mdlWidth = mdlInfo.get_width();
		uint32_t mdlHeight = mdlInfo.get_height();

		finalScaleWidth = mdlWidth / (inputWidthOverride * 1.f);
		finalScaleHeight = mdlHeight / (inputHeightOverride * 1.f);
	}

	finalScaleWidth = inputScaleWidth;
	finalScaleHeight = inputScaleHeight;

	for (object_proposed& obj : objects)
	{
		float x0 = (obj.x) / finalScaleWidth;
		float y0 = (obj.y) / finalScaleHeight;
		float x1 = (obj.x + obj.w) / finalScaleWidth;
		float y1 = (obj.y + obj.h) / finalScaleHeight;

		obj.x = x0;
		obj.y = y0;
		obj.w = x1 - x0;
		obj.h = y1 - y0;
	}
}



void yolox_inference::append_executor()
{
}



bool yolox_inference::create_session()
{
	if (!ortApi || !ortEnv || !ortSessionOptions)
	{
		return false;
	}

	ortApi->CreateSession(ortEnv, mdlInfo.get_path().c_str(), ortSessionOptions, &ortSession);
	if (!ortSession)
	{
		return false;
	}

	ortApi->SessionGetInputName(ortSession, 0, ortAllocator, &mdlInputName);
	ortApi->SessionGetOutputName(ortSession, 0, ortAllocator, &mdlOutputName);
	isReady = true;

	return true;
}

bool yolox_inference::inference_float(float* pixelData, uint32_t width, uint32_t height)
{
	if (!isReady)
	{
		return false;
	}

	uint32_t mdlWidth = mdlInfo.get_width();
	uint32_t mdlHeight = mdlInfo.get_height();
	uint32_t mdlNumPixels = mdlWidth * mdlHeight;
	uint8_t mdlChannels = model_info::modelChannels;

	std::vector<float>& mdlNorm = mdlInfo.get_norm();
	std::vector<float>& mdlMean = mdlInfo.get_mean();
	size_t minNormalizeChannels = std::min(mdlNorm.size(), mdlMean.size());
	bool shouldNormalize = (minNormalizeChannels > 0);

	if (shouldNormalize && minNormalizeChannels != mdlChannels)
	{
		return false;
	}

	bool shouldResize = (width != mdlWidth || height != mdlHeight);
	float* pixelDataFinal = shouldResize ? inputData : pixelData;

	inputWidth = width;
	inputHeight = height;

	inputScaleWidth = mdlWidth / (inputWidth * 1.f);
	inputScaleHeight = mdlHeight / (inputHeight * 1.f);
	
	if (shouldResize)
	{
#ifdef YOLOX_USE_STB_RESIZE2
		for (uint8_t c = 0; c < mdlChannels; ++c)
		{
			float* inputChannel = pixelData + c * width * height;
			float* outputChannel = inputData + c * mdlWidth * mdlHeight;

			stbir_resize_float_linear(
				inputChannel, width, height, width * sizeof(float),
				outputChannel, mdlWidth, mdlHeight, mdlWidth * sizeof(float),
				STBIR_1CHANNEL
			);
		}
#else
		return false;
#endif
	}

	if (shouldNormalize || !shouldResize)
	{
		for (uint32_t px = 0; px < mdlNumPixels; px++)
		{
			for (uint8_t c = 0; c < mdlChannels; c++)
			{
				float chanValue = pixelDataFinal[c * mdlNumPixels + px];
				if (shouldNormalize)
				{
					chanValue = (chanValue - mdlMean[c]) / mdlNorm[c];
				}

				inputData[c * mdlNumPixels + px] = chanValue;
			}
		}
	}
	
	OrtMemoryInfo* ortMemInfo = nullptr;
	if (ortApi->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ortMemInfo))
	{
		return false;
	}

	OrtValue* inputTensor = nullptr;
	int64_t inputShape[] = { 1, 3, mdlHeight, mdlWidth };
	size_t inputShapeLen = sizeof(inputShape) / sizeof(inputShape[0]);

	OrtStatusPtr tensorCreated = ortApi->CreateTensorWithDataAsOrtValue(
		ortMemInfo, inputData, mdlWidth * mdlHeight * sizeof(float) * mdlChannels,
		inputShape, inputShapeLen, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputTensor
	);

	if (tensorCreated)
	{
		return false;
	}

	OrtTensorTypeAndShapeInfo* outputTensorInfo;
	OrtValue* outputTensor = nullptr;
	size_t outputTensorLength = 0;
	int64_t outputTensorDims[3] = {};

	const char* inputNames[] = { mdlInputName };
	const char* outputNames[] = { mdlOutputName };

	if (ortApi->Run(ortSession, nullptr, inputNames, (const OrtValue* const*)&inputTensor, 1, outputNames, 1, &outputTensor))
	{
		ortApi->ReleaseValue(inputTensor);
		return false;
	}

	if (!outputTensor)
	{
		ortApi->ReleaseValue(inputTensor);
		ortApi->ReleaseValue(outputTensor);
		return false;
	}

	ortApi->GetTensorTypeAndShape(outputTensor, &outputTensorInfo);
	ortApi->GetDimensionsCount(outputTensorInfo, &outputTensorLength);
	ortApi->GetDimensions(outputTensorInfo, outputTensorDims, outputTensorLength);

	float* outputTensorData = nullptr;
	ortApi->GetTensorMutableData(outputTensor, (void**)&outputTensorData);

	process_proposals(outputTensorData, outputTensorDims[2]);
	nms_objects();
	scale_objects();

	ortApi->ReleaseValue(inputTensor);
	ortApi->ReleaseValue(outputTensor);
	return true;
}

bool yolox_inference::inference_rgbx8888(uint8_t* pixelData, uint32_t width, uint32_t height)
{
	if (!isReady)
	{
		return false;
	}

	uint8_t numChannels = model_info::modelChannels;
	uint32_t planarNumPixels = width * height;
	float* planarPixels = new float[planarNumPixels * numChannels];
	
	for (uint32_t px = 0; px < planarNumPixels; px++)
	{
		for (uint8_t c = 0; c < numChannels; c++)
		{
			float chanValue = static_cast<float>(pixelData[px * 4 + c]) / 255;
			planarPixels[c * planarNumPixels + px] = chanValue;
		}
	}
	
	bool success = false;
	
	if (planarPixels != nullptr)
	{
		success = inference_float(planarPixels, width, height);
		delete[] planarPixels;
	}

	return success;
}

std::vector<yolox_inference::object_proposed>& yolox_inference::get_objects()
{
	return objects;
}

float yolox_inference::get_probability_threshold()
{
	return probThreshold;
}

float yolox_inference::get_nms_threshold()
{
	return nmsThreshold;
}

uint32_t yolox_inference::get_input_width()
{
	return inputWidth;
}

uint32_t yolox_inference::get_input_height()
{
	return inputHeight;
}

void yolox_inference::set_probability_threshold(float threshold)
{
	probThreshold = threshold;
}

void yolox_inference::set_nms_threshold(float threshold)
{
	nmsThreshold = threshold;
}

void yolox_inference::set_scale_override(bool enabled, uint32_t width, uint32_t height)
{
	scaleOverride = enabled;
	inputWidthOverride = width;
	inputHeightOverride = height;
}

yolox_inference::model_info& yolox_inference::get_model_info()
{
	return mdlInfo;
}



#ifdef YOLOX_INFERENCE_USE_DML
yolox_inference_dml::yolox_inference_dml(uint8_t devId, std::wstring modelPath, uint32_t modelWidth, uint32_t modelHeight):
	yolox_inference(devId, modelPath, modelWidth, modelHeight)
{
}

void yolox_inference_dml::append_executor()
{
	ortApi->SetSessionGraphOptimizationLevel(ortSessionOptions, ORT_ENABLE_BASIC);
	ortApi->DisableMemPattern(ortSessionOptions);
	ortApi->SetSessionExecutionMode(ortSessionOptions, ExecutionMode::ORT_SEQUENTIAL);
	OrtSessionOptionsAppendExecutionProvider_DML(ortSessionOptions, deviceId);
}
#endif