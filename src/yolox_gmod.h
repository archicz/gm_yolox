#ifndef YOLOX_GMOD_H
#define YOLOX_GMOD_H

#include "common.h"
#include "yolox_inference.h"
#include "materialsystem.h"
#include "luashared.h"
#include <atomic>
#include <queue>

class yolox_gmod : public yolox_inference_dml
{
public:
	struct rt_data
	{
		uint8_t* pixels;
		uint32_t width;
		uint32_t height;
	};
private:
	HANDLE inferenceThread;
	HANDLE queueMutex;
	
	std::atomic<bool> dataReady;
	std::queue<rt_data> queuedRT;
	std::vector<object_proposed> objectsThreaded;
	std::vector<object_proposed> objectsCopy;
public:
	yolox_gmod(std::wstring modelPath, uint32_t modelWidth, uint32_t modelHeight);
	virtual ~yolox_gmod() override;
public:
	bool inference_rendertarget(ITexture* rt);
	std::vector<object_proposed>& get_objects_threaded();
	static void push_objects_lua(std::vector<object_proposed>& objects, ILuaBase* LUA);
private:
	static DWORD inference_thread(LPVOID parameter);
	void add_queue(uint8_t* pixelData, uint32_t width, uint32_t height);
};

#endif