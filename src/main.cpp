#include "common.h"
#include "materialsystem.h"
#include "luashared.h"
#include "yolox_gmod.h"

yolox_gmod* yoloxInference = nullptr;
IMaterialSystem* materialSystem = nullptr;

LUA_FUNCTION(Initialize)
{
    if (yoloxInference != nullptr)
    {
        delete yoloxInference;
        yoloxInference = nullptr;
    }

    const char* modelPath = LUA->GetString(1);
    uint32_t modelWidth = static_cast<uint32_t>(LUA->GetNumber(2));
    uint32_t modelHeight = static_cast<uint32_t>(LUA->GetNumber(3));

#ifdef GMOD_WINDOWS
    int modelPathSize = MultiByteToWideChar(CP_UTF8, 0, modelPath, -1, NULL, 0);
    wchar_t* modelPathWide = new wchar_t[modelPathSize];
    MultiByteToWideChar(CP_UTF8, 0, modelPath, -1, modelPathWide, modelPathSize);

    yoloxInference = new yolox_gmod(modelPathWide, modelWidth, modelHeight);
    delete[] modelPathWide;
#endif
    
    return 1;
}

LUA_FUNCTION(SetNormal)
{
    if (!yoloxInference)
    {
        LUA->ThrowError("Not initialized");
        return 0;
    }

    int numArgs = LUA->Top();
    std::vector<float> normals;

    for (int i = 1; i < numArgs + 1; i++)
    {
        normals.push_back(static_cast<float>(LUA->GetNumber(i)));
    }

    yolox_inference::model_info& mdl = yoloxInference->get_model_info();
    mdl.set_norm(normals);

    return 1;
}

LUA_FUNCTION(SetMean)
{
    if (!yoloxInference)
    {
        LUA->ThrowError("Not initialized");
        return 0;
    }

    int numArgs = LUA->Top();
    std::vector<float> means;

    for (int i = 1; i < numArgs + 1; i++)
    {
        means.push_back(static_cast<float>(LUA->GetNumber(i)));
    }

    yolox_inference::model_info& mdl = yoloxInference->get_model_info();
    mdl.set_mean(means);

    return 1;
}

LUA_FUNCTION(SetProbabilityThreshold)
{
    if (!yoloxInference)
    {
        LUA->ThrowError("Not initialized");
        return 0;
    }
    
    float prob = static_cast<float>(LUA->CheckNumber(1));
    yoloxInference->set_probability_threshold(prob);

    return 1;
}

LUA_FUNCTION(SetNMSThreshold)
{
    if (!yoloxInference)
    {
        LUA->ThrowError("Not initialized");
        return 0;
    }

    float nms = static_cast<float>(LUA->CheckNumber(1));
    yoloxInference->set_nms_threshold(nms);

    return 1;
}

LUA_FUNCTION(CreateSession)
{
    if (!yoloxInference)
    {
        LUA->ThrowError("Not initialized");
        return 0;
    }

    LUA->PushBool(yoloxInference->create_session());

    return 1;
}

LUA_FUNCTION(AddRenderTarget)
{
    if (!yoloxInference)
    {
        LUA->ThrowError("Not initialized");
        return 0;
    }

    if (!LUA->IsType(1, kTexture))
    {
        LUA->ThrowError("Invalid render target");
        return 0;
    }

    ITexture* rt = reinterpret_cast<ITexture*>(LUA->GetUserdata(1)->data);
    yoloxInference->inference_rendertarget(rt);
    
    return 1;
}

LUA_FUNCTION(SetScaleOverride)
{
    if (!yoloxInference)
    {
        LUA->ThrowError("Not initialized");
        return 0;
    }

    bool enabled = LUA->GetBool(1);
    uint32_t w = static_cast<uint32_t>(LUA->CheckNumber(2));
    uint32_t h = static_cast<uint32_t>(LUA->CheckNumber(3));

    yoloxInference->set_scale_override(enabled, w, h);

    return 1;
}

LUA_FUNCTION(GetObjects)
{
    if (!yoloxInference)
    {
        LUA->ThrowError("Not initialized");
        return 0;
    }

    auto& objects = yoloxInference->get_objects_threaded();
    yolox_gmod::push_objects_lua(objects, LUA);

    return 1;
}

GMOD_MODULE_OPEN()
{
#ifdef GMOD_WINDOWS
    materialSystem = (IMaterialSystem*)GetInterface("materialsystem.dll", MATERIAL_SYSTEM_INTERFACE_VERSION);
#endif
    if (!materialSystem)
    {
        LUA->ThrowError("Failed to acquire materialsystem interface!");
        return 1;
    }

    LUA->PushSpecial(SPECIAL_GLOB);
    LUA->CreateTable();
        LUA->PushCFunction(Initialize);
        LUA->SetField(-2, "Initialize");

        LUA->PushCFunction(SetNormal);
        LUA->SetField(-2, "SetNormal");

        LUA->PushCFunction(SetMean);
        LUA->SetField(-2, "SetMean");

        LUA->PushCFunction(SetNMSThreshold);
        LUA->SetField(-2, "SetNMSThreshold");

        LUA->PushCFunction(SetProbabilityThreshold);
        LUA->SetField(-2, "SetProbabilityThreshold");

        LUA->PushCFunction(CreateSession);
        LUA->SetField(-2, "CreateSession");
        
        LUA->PushCFunction(AddRenderTarget);
        LUA->SetField(-2, "AddRenderTarget");

        LUA->PushCFunction(SetScaleOverride);
        LUA->SetField(-2, "SetScaleOverride");

        LUA->PushCFunction(GetObjects);
        LUA->SetField(-2, "GetObjects");
    LUA->SetField(-2, "YOLOX");
    LUA->Pop();

    return 0;
}

GMOD_MODULE_CLOSE()
{
    if (yoloxInference != nullptr)
    {
        delete yoloxInference;
        yoloxInference = nullptr;
    }

    return 0;
}