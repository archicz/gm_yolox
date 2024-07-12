#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#include "yolox_gmod.h"

DWORD yolox_gmod::inference_thread(LPVOID parameter)
{
    while (true)
    {
        yolox_gmod* inference = reinterpret_cast<yolox_gmod*>(parameter);
        HANDLE queueMutex = inference->queueMutex;
        auto& queuedRT = inference->queuedRT;
        auto& dataReady = inference->dataReady;

        WaitForSingleObject(queueMutex, INFINITE);
        if (queuedRT.size() > 0)
        {
            rt_data& frontData = queuedRT.front();

            inference->inference_rgbx8888(frontData.pixels, frontData.width, frontData.height);
            delete[] frontData.pixels;

            queuedRT.pop();
        }
        ReleaseMutex(queueMutex);

        auto& objects = inference->get_objects();
        if (!dataReady.load(std::memory_order_acquire))
        {
            auto& objectsThreaded = inference->objectsThreaded;
            objectsThreaded.assign(objects.begin(), objects.end());
            dataReady.store(true, std::memory_order_release);
        }
    }

    return 0;
}

yolox_gmod::yolox_gmod(std::wstring modelPath, uint32_t modelWidth, uint32_t modelHeight):
    yolox_inference_dml(0, modelPath, modelWidth, modelHeight),
    inferenceThread(nullptr), queueMutex(nullptr), dataReady(false)
{
    inferenceThread = CreateThread(NULL, NULL, (LPTHREAD_START_ROUTINE)yolox_gmod::inference_thread, this, NULL, NULL);
    queueMutex = CreateMutexA(NULL, false, NULL);
}

yolox_gmod::~yolox_gmod()
{
    TerminateThread(inferenceThread, 0);
    CloseHandle(queueMutex);
}

std::vector<yolox_inference::object_proposed>& yolox_gmod::get_objects_threaded()
{
    if (dataReady.load(std::memory_order_acquire))
    {
        objectsCopy.assign(objectsThreaded.begin(), objectsThreaded.end());
        dataReady.store(false, std::memory_order_release);
    }
    
    return objectsCopy;
}

void yolox_gmod::push_objects_lua(std::vector<object_proposed>& objects, ILuaBase* LUA)
{
    LUA->CreateTable();

    if (objects.size() == 0)
    {
        return;
    }

    for (size_t i = 0; i < objects.size(); i++)
    {
        auto& object = objects.at(i);

        double id = static_cast<double>(i + 1);
        double label = static_cast<double>(object.label);
        double prob = static_cast<double>(object.prob);
        double x = floor(static_cast<double>(object.x));
        double y = floor(static_cast<double>(object.y));
        double w = floor(static_cast<double>(object.w));
        double h = floor(static_cast<double>(object.h));

        LUA->PushNumber(id);
        LUA->CreateTable();

        LUA->PushString("label");
        LUA->PushNumber(label);
        LUA->SetTable(-3);

        LUA->PushString("prob");
        LUA->PushNumber(prob);
        LUA->SetTable(-3);

        LUA->PushString("x");
        LUA->PushNumber(x);
        LUA->SetTable(-3);

        LUA->PushString("y");
        LUA->PushNumber(y);
        LUA->SetTable(-3);

        LUA->PushString("w");
        LUA->PushNumber(w);
        LUA->SetTable(-3);

        LUA->PushString("h");
        LUA->PushNumber(h);
        LUA->SetTable(-3);

        LUA->SetTable(-3);
    }
}

bool yolox_gmod::inference_rendertarget(ITexture* rt)
{
    if (!isReady)
    {
        return false;
    }

    IMatRenderContext* renderCtx = materialSystem->GetRenderContext();
    uint32_t rtWidth = 0;
    uint32_t rtHeight = 0;
    uint32_t rtByteSize = 0;
    uint8_t* rtPixels = nullptr;

    renderCtx->BeginRender();
        renderCtx->Flush(false);
        renderCtx->SetRenderTarget(rt);
        renderCtx->GetRenderTargetDimensions(&rtWidth, &rtHeight);

        rtByteSize = 4 * rtWidth * rtHeight;
        rtPixels = new uint8_t[rtByteSize];

        renderCtx->ReadPixels(0, 0, rtWidth, rtHeight, rtPixels, IMAGE_FORMAT_RGBA8888);
        renderCtx->SetRenderTarget(nullptr);
    renderCtx->EndRender();
    
    if (rtPixels != nullptr)
    {
        add_queue(rtPixels, rtWidth, rtHeight);
    }
    
    return (rtPixels != nullptr);
}

void yolox_gmod::add_queue(uint8_t* pixelData, uint32_t width, uint32_t height)
{
    rt_data queueData;
    queueData.pixels = pixelData;
    queueData.width = width;
    queueData.height = height;

    WaitForSingleObject(queueMutex, INFINITE);
    queuedRT.push(queueData);
    ReleaseMutex(queueMutex);
}