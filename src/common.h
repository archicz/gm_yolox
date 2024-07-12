#ifndef COMMON_H
#define COMMON_H
#pragma once

#if defined(_WIN32) || defined(_WIN64)
#define GMOD_WINDOWS
#ifdef _WIN64
#define GMOD_WIN_64
#else
#define GMOD_WIN_32
#endif
#endif

#ifdef GMOD_WINDOWS
#define _CRT_SECURE_NO_WARNINGS
#include <Windows.h>
#endif

#include <stdio.h>
#include <cstdint>

template<typename Fn>
inline Fn CallVFunction(void* ppClass, size_t index)
{
	uintptr_t* pVTable = *(uintptr_t**)ppClass;
	uintptr_t dwAddress = pVTable[index];
	return reinterpret_cast<Fn>(dwAddress);
}

inline void* GetExport(const char* moduleName, const char* exportName)
{
#ifdef GMOD_WINDOWS
	HMODULE moduleDll = GetModuleHandleA(moduleName);
	if (!moduleDll)
	{
		return nullptr;
	}

	return GetProcAddress(moduleDll, exportName);
#endif
}

typedef void* (*CreateInterfaceFn)(const char* interfaceName, void* returnCode);
inline void* GetInterface(const char* moduleName, const char* version)
{
	CreateInterfaceFn factory = (CreateInterfaceFn)GetExport(moduleName, "CreateInterface");
	if (!factory)
	{
		return nullptr;
	}

	return factory(version, 0);
}

#endif