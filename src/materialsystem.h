#ifndef MATERIALSYSTEM_H
#define MATERIALSYSTEM_H
#pragma once

#include "common.h"

#ifdef GMOD_WINDOWS
#define MATERIAL_SYSTEM_INTERFACE_VERSION "VMaterialSystem080"
#define MATERIAL_SYSTEM_GETRENDERCONTEXT_INDEX 102

#define MATERIAL_RENDERCONTEXT_READPIXELS_INDEX 11
#define MATERIAL_RENDERCONTEXT_SETRENDERTARGET_INDEX 4
#define MATERIAL_RENDERCONTEXT_GETRENDERTARGETDIMENSIONS_INDEX 6
#endif

class ITexture;

enum ImageFormat
{
	IMAGE_FORMAT_UNKNOWN = -1,
	IMAGE_FORMAT_RGBA8888 = 0,
	IMAGE_FORMAT_ABGR8888,
	IMAGE_FORMAT_RGB888,
	IMAGE_FORMAT_BGR888,
	IMAGE_FORMAT_RGB565,
	IMAGE_FORMAT_I8,
	IMAGE_FORMAT_IA88,
	IMAGE_FORMAT_P8,
	IMAGE_FORMAT_A8,
	IMAGE_FORMAT_RGB888_BLUESCREEN,
	IMAGE_FORMAT_BGR888_BLUESCREEN,
	IMAGE_FORMAT_ARGB8888,
	IMAGE_FORMAT_BGRA8888,
};

class IMatRenderContext
{
public:
	virtual void BeginRender() = 0;
	virtual void EndRender() = 0;
	virtual void Flush(bool unknown) = 0;
public:
#ifdef GMOD_WIN_64
	typedef int(__cdecl* ReadPixelsFn)(void*, uint32_t, uint32_t, uint32_t, uint32_t, uint8_t*, ImageFormat);
#endif

#ifdef GMOD_WIN_32
	typedef int(__thiscall* ReadPixelsFn)(void*, uint32_t, uint32_t, uint32_t, uint32_t, uint8_t*, ImageFormat);
#endif
	int ReadPixels(uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t* data, ImageFormat format)
	{
		return CallVFunction<ReadPixelsFn>(this, MATERIAL_RENDERCONTEXT_READPIXELS_INDEX)
			(this, x, y, width, height, data, format);
	}
	
#ifdef GMOD_WIN_64
	typedef int(__cdecl* SetRenderTargetFn)(void*, ITexture*);
#endif

#ifdef GMOD_WIN_32
	typedef int(__thiscall* SetRenderTargetFn)(void*, ITexture*);
#endif
	int SetRenderTarget(ITexture* rt)
	{
		return CallVFunction<SetRenderTargetFn>(this, MATERIAL_RENDERCONTEXT_SETRENDERTARGET_INDEX)
			(this, rt);
	}
	
#ifdef GMOD_WIN_64
	typedef int(__cdecl* GetRenderTargetDimensionsFn)(void*, uint32_t*, uint32_t*);
#endif

#ifdef GMOD_WIN_32
	typedef int(__thiscall* GetRenderTargetDimensionsFn)(void*, uint32_t*, uint32_t*);
#endif
	int GetRenderTargetDimensions(uint32_t* width, uint32_t* height)
	{
		return CallVFunction<GetRenderTargetDimensionsFn>(this, MATERIAL_RENDERCONTEXT_GETRENDERTARGETDIMENSIONS_INDEX)
			(this, width, height);
	}
};

class IMaterialSystem
{
public:
	typedef IMatRenderContext* (__thiscall* GetRenderContextFn)(void*);
	IMatRenderContext* GetRenderContext()
	{
		return CallVFunction<GetRenderContextFn>(this, MATERIAL_SYSTEM_GETRENDERCONTEXT_INDEX)
			(this);
	}
};

extern IMaterialSystem* materialSystem;

#endif