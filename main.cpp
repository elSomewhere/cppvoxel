// Minimal WebGPU triangle using Emscripten
// Ultra-simplified version for maximum compatibility

#include <emscripten.h>
#include <emscripten/html5.h>
#include <webgpu/webgpu.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Global variables
static WGPUDevice device = NULL;
static WGPUQueue queue = NULL;
static WGPUSwapChain swapChain = NULL;
static WGPURenderPipeline pipeline = NULL;
static int frameCount = 0;
static bool initialized = false;
static bool hasError = false;

// The simplest possible triangle shader
static const char* shaderSource = R"(
// Vertex shader
@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32) -> @builtin(position) vec4f {
    var positions = array<vec2f, 3>(
        vec2f(0.0, 0.5),    // top center
        vec2f(-0.5, -0.5),  // bottom left
        vec2f(0.5, -0.5)    // bottom right
    );
    return vec4f(positions[vertex_index], 0.0, 1.0);
}

// Fragment shader - just returns a solid color
@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 1.0); // Solid red
}
)";

// JavaScript console.log wrapper
EM_JS(void, js_console_log, (const char* str), {
    console.log(UTF8ToString(str));
});

// Logs a message to both console and JavaScript
void LogMessage(const char* message) {
    printf("%s\n", message);
    js_console_log(message);
}

// Draw a single frame
void DrawFrame() {
    if (!initialized || !device || !swapChain || !pipeline || hasError) {
        return;
    }
    
    // Get the next texture to render to
    WGPUTextureView view = wgpuSwapChainGetCurrentTextureView(swapChain);
    if (!view) {
        LogMessage("Failed to get next texture view");
        return;
    }
    
    // Create a command encoder
    WGPUCommandEncoderDescriptor encoderDesc = {};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encoderDesc);
    if (!encoder) {
        wgpuTextureViewRelease(view);
        LogMessage("Failed to create command encoder");
        return;
    }
    
    // Begin a render pass
    WGPUColor clearColor = {0.0f, 0.0f, 0.1f, 1.0f};
    WGPURenderPassColorAttachment colorAttachment = {};
    colorAttachment.view = view;
            colorAttachment.loadOp = WGPULoadOp_Clear;
            colorAttachment.storeOp = WGPUStoreOp_Store;
    colorAttachment.clearValue = clearColor;

    WGPURenderPassDescriptor renderPassDesc = {};
            renderPassDesc.colorAttachmentCount = 1;
            renderPassDesc.colorAttachments = &colorAttachment;
            
            WGPURenderPassEncoder renderPass = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDesc);
    if (!renderPass) {
        wgpuCommandEncoderRelease(encoder);
        wgpuTextureViewRelease(view);
        LogMessage("Failed to begin render pass");
        return;
    }

    // Draw the triangle
    wgpuRenderPassEncoderSetPipeline(renderPass, pipeline);
    wgpuRenderPassEncoderDraw(renderPass, 3, 1, 0, 0);
            wgpuRenderPassEncoderEnd(renderPass);

    // Finish and submit
    WGPUCommandBufferDescriptor cmdBufferDesc = {};
    WGPUCommandBuffer cmdBuffer = wgpuCommandEncoderFinish(encoder, &cmdBufferDesc);
    if (!cmdBuffer) {
            wgpuRenderPassEncoderRelease(renderPass);
        wgpuCommandEncoderRelease(encoder);
        wgpuTextureViewRelease(view);
        LogMessage("Failed to finish command buffer");
        return;
    }

    wgpuQueueSubmit(queue, 1, &cmdBuffer);
        
        // Cleanup
    wgpuCommandBufferRelease(cmdBuffer);
    wgpuRenderPassEncoderRelease(renderPass);
        wgpuCommandEncoderRelease(encoder);
    wgpuTextureViewRelease(view);
}

// Error callback
void OnError(WGPUErrorType type, const char* message, void*) {
    char buffer[1024];
    snprintf(buffer, sizeof(buffer), "WebGPU Error: %s", message);
    LogMessage(buffer);
    hasError = true;
}

// Device error callback
void OnDeviceError(WGPUErrorType type, const char* message, void*) {
    char buffer[1024];
    snprintf(buffer, sizeof(buffer), "Device Error: %s", message);
    LogMessage(buffer);
    hasError = true;
}

// Create the rendering pipeline
bool CreateRenderPipeline() {
    LogMessage("Creating render pipeline...");

    // Create swap chain
    WGPUSwapChainDescriptor swapChainDesc = {};
        swapChainDesc.usage = WGPUTextureUsage_RenderAttachment;
        swapChainDesc.format = WGPUTextureFormat_BGRA8Unorm;
    swapChainDesc.width = 800;
    swapChainDesc.height = 600;
        swapChainDesc.presentMode = WGPUPresentMode_Fifo;
        
    WGPUSurfaceDescriptorFromCanvasHTMLSelector canvasDesc = {};
    canvasDesc.chain.sType = WGPUSType_SurfaceDescriptorFromCanvasHTMLSelector;
    canvasDesc.selector = "#canvas";
    
    WGPUSurfaceDescriptor surfaceDesc = {};
    surfaceDesc.nextInChain = (WGPUChainedStruct*)&canvasDesc;
    
    WGPUSurface surface = wgpuInstanceCreateSurface(wgpuCreateInstance(NULL), &surfaceDesc);
    if (!surface) {
        LogMessage("Failed to create surface");
        return false;
    }
    
    swapChain = wgpuDeviceCreateSwapChain(device, surface, &swapChainDesc);
    if (!swapChain) {
        LogMessage("Failed to create swap chain");
        return false;
    }

    // Create shader module
    WGPUShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    wgslDesc.code = shaderSource;
    
    WGPUShaderModuleDescriptor shaderDesc = {};
    shaderDesc.nextInChain = (const WGPUChainedStruct*)&wgslDesc;
    
    WGPUShaderModule shaderModule = wgpuDeviceCreateShaderModule(device, &shaderDesc);
    if (!shaderModule) {
        LogMessage("Failed to create shader module");
        return false;
    }
    
    // Setup fragment state with one color target
    WGPUBlendState blend = {};
    blend.color.srcFactor = WGPUBlendFactor_One;
    blend.color.dstFactor = WGPUBlendFactor_Zero;
    blend.color.operation = WGPUBlendOperation_Add;
    blend.alpha.srcFactor = WGPUBlendFactor_One;
    blend.alpha.dstFactor = WGPUBlendFactor_Zero;
    blend.alpha.operation = WGPUBlendOperation_Add;

    WGPUColorTargetState colorTarget = {};
        colorTarget.format = WGPUTextureFormat_BGRA8Unorm;
    colorTarget.blend = &blend;
        colorTarget.writeMask = WGPUColorWriteMask_All;
        
    WGPUFragmentState fragment = {};
    fragment.module = shaderModule;
    fragment.entryPoint = "fs_main";
    fragment.targetCount = 1;
    fragment.targets = &colorTarget;

    // Create a pipeline layout
    WGPUPipelineLayoutDescriptor layoutDesc = {};
    WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &layoutDesc);
    if (!pipelineLayout) {
        wgpuShaderModuleRelease(shaderModule);
        LogMessage("Failed to create pipeline layout");
        return false;
    }
    
    // Create the render pipeline
    WGPURenderPipelineDescriptor pipelineDesc = {};
    pipelineDesc.layout = pipelineLayout;
    
    // Vertex state (minimal configuration for vertexIndex-based vertices)
    pipelineDesc.vertex.module = shaderModule;
    pipelineDesc.vertex.entryPoint = "vs_main";
    
    // Fragment stage
    pipelineDesc.fragment = &fragment;
    
    // Primitive assembly 
    pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    pipelineDesc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    pipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
    pipelineDesc.primitive.cullMode = WGPUCullMode_None;
    
    // Add multisample state - this is required (default of 0 is invalid)
    pipelineDesc.multisample.count = 1;
    pipelineDesc.multisample.mask = 0xFFFFFFFF;
    pipelineDesc.multisample.alphaToCoverageEnabled = false;
    
    pipeline = wgpuDeviceCreateRenderPipeline(device, &pipelineDesc);
    
    // Cleanup
    wgpuShaderModuleRelease(shaderModule);
    wgpuPipelineLayoutRelease(pipelineLayout);
    
    if (!pipeline) {
        LogMessage("Failed to create render pipeline");
        return false;
    }
    
    LogMessage("Render pipeline created successfully");
    return true;
}

// Device creation callback
void OnDeviceCreated(WGPURequestDeviceStatus status, WGPUDevice dev, const char* message, void*) {
    if (status != WGPURequestDeviceStatus_Success) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "Failed to create device: %s", message ? message : "unknown error");
        LogMessage(buffer);
        hasError = true;
        return;
    }
    
    LogMessage("Device created successfully");
    device = dev;
    queue = wgpuDeviceGetQueue(device);
    
    // Set error callback
    wgpuDeviceSetUncapturedErrorCallback(device, OnDeviceError, NULL);
    
    // Create the render pipeline
    if (CreateRenderPipeline()) {
        initialized = true;
    } else {
        hasError = true;
    }
}

// Adapter creation callback
void OnAdapterReady(WGPURequestAdapterStatus status, WGPUAdapter adapter, const char* message, void* userdata) {
    if (status != WGPURequestAdapterStatus_Success) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "Failed to get adapter: %s", message ? message : "unknown error");
        LogMessage(buffer);
        hasError = true;
        return;
    }

    LogMessage("Adapter found, requesting device");

    // Request a device with default settings (no custom limits)
    WGPUDeviceDescriptor deviceDesc = {};
    wgpuAdapterRequestDevice(adapter, &deviceDesc, OnDeviceCreated, NULL);
}

// Initialize WebGPU
void InitWebGPU() {
    LogMessage("Initializing WebGPU...");

    // Initialize instance with default descriptor
    WGPUInstanceDescriptor desc = {};
    WGPUInstance instance = wgpuCreateInstance(&desc);
    if (!instance) {
        LogMessage("Failed to create WebGPU instance");
        hasError = true;
        return;
    }

    // Request adapter with default options
    WGPURequestAdapterOptions adapterOpts = {};
    wgpuInstanceRequestAdapter(instance, &adapterOpts, OnAdapterReady, NULL);
}

// Main loop function called by Emscripten
void MainLoop() {
    frameCount++;
    
    // Try to render a frame
    DrawFrame();
    
    // Debug output every 60 frames
    if (frameCount % 60 == 0) {
        char buffer[128];
        snprintf(buffer, sizeof(buffer), "Frame: %d, Initialized: %s, Has Error: %s", 
               frameCount, 
               initialized ? "Yes" : "No",
               hasError ? "Yes" : "No");
        LogMessage(buffer);
    }
}

// Main entry point
int main() {
    LogMessage("Starting Ultra-minimal WebGPU Triangle");
    
    // Initialize WebGPU
    InitWebGPU();
    
    // Start the main loop
    emscripten_set_main_loop(MainLoop, 0, 1);
    
    return 0;
}