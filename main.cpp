// WebGPU Voxel Engine - Built on minimal Emscripten WebGPU example
// Focuses on GPU-managed voxels with instancing, octree spatial organization, and efficient culling

#include <emscripten.h>
#include <emscripten/html5.h>
#include <webgpu/webgpu.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <vector>
#include <array>
#include <memory>
#include <functional>
#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <random>
#include <chrono>

// Constants
constexpr uint32_t MAX_VOXELS = 65536;  // Max number of voxels
constexpr uint32_t OCTREE_MAX_DEPTH = 8; // Max depth of octree
constexpr float WORLD_SIZE = 100.0f;    // World size
constexpr uint32_t WORKGROUP_SIZE = 64; // Compute shader workgroup size
constexpr uint32_t CANVAS_WIDTH = 800;  // Canvas width
constexpr uint32_t CANVAS_HEIGHT = 600; // Canvas height

// Forward declarations
struct Voxel;
struct OctreeNode;
class VoxelOctree;
class VoxelSystem;
class GPUVoxelManager;

// Global variables
static WGPUDevice device = nullptr;
static WGPUQueue queue = nullptr;
static WGPUSwapChain swapChain = nullptr;
static WGPURenderPipeline renderPipeline = nullptr;
static WGPUComputePipeline computePipeline = nullptr;
static WGPUBindGroup renderBindGroup = nullptr;
static WGPUBindGroup computeBindGroup = nullptr;
static WGPUBindGroupLayout bindGroupLayout = nullptr;
static WGPUBuffer cameraBuffer = nullptr;
static WGPUBuffer voxelBuffer = nullptr;
static WGPUBuffer visibleIndicesBuffer = nullptr;
static WGPUBuffer visibleCountBuffer = nullptr;
static WGPUTexture depthTexture = nullptr;
static WGPUTextureView depthTextureView = nullptr;
static int frameCount = 0;
static bool initialized = false;
static bool hasError = false;

// Our voxel engine systems
static std::unique_ptr<VoxelSystem> voxelSystem;
static std::unique_ptr<GPUVoxelManager> gpuVoxelManager;

// JavaScript console.log wrapper
EM_JS(void, js_console_log, (const char* str), {
    console.log(UTF8ToString(str));
});

// Logs a message to both console and JavaScript
void LogMessage(const char* message) {
    printf("%s\n", message);
    js_console_log(message);
}

// Voxel Structure
struct Voxel {
    float x, y, z;     // Position
    uint32_t color;    // RGBA color (8 bits per channel)
    uint8_t voxelType; // Voxel type (0: static, 1: floating, 2: orbiting)
    bool isActive;     // Is voxel active?

    Voxel() : x(0.0f), y(0.0f), z(0.0f), color(0xFFFFFFFF), voxelType(0), isActive(true) {}
    
    Voxel(float _x, float _y, float _z, uint32_t _color, uint8_t _voxelType, bool _isActive) 
        : x(_x), y(_y), z(_z), color(_color), voxelType(_voxelType), isActive(_isActive) {}
};

// GPU Buffer management
struct GPUBufferInfo {
    WGPUBuffer buffer;
    size_t size;
    size_t capacity;
    bool needsUpdate;
};

// Camera data for shaders
struct CameraData {
    float viewProjection[16];  // View-projection matrix (column-major)
    float cameraPosition[3];   // Camera position
    float padding;             // For alignment
    float frustumPlanes[6][4]; // Frustum planes (normal.xyz, distance)
    float maxDrawDistance;     // Maximum draw distance
};

// Octree Node Structure
struct OctreeNode {
    bool hasChildren;
    bool isLeaf;
    uint32_t voxelCount;
    float centerX, centerY, centerZ;
    float size;
    std::array<std::shared_ptr<OctreeNode>, 8> children;
    std::vector<uint32_t> voxelIndices; // Indices into the global voxel array
    
    OctreeNode(float cx, float cy, float cz, float s) 
        : hasChildren(false), isLeaf(true), voxelCount(0), 
          centerX(cx), centerY(cy), centerZ(cz), size(s) {
        children.fill(nullptr);
    }
};

// Octree for Spatial Organization
class VoxelOctree {
public:
    VoxelOctree(float centerX, float centerY, float centerZ, float size, uint32_t maxDepth)
        : m_centerX(centerX), m_centerY(centerY), m_centerZ(centerZ), 
          m_size(size), m_maxDepth(maxDepth) {
        m_root = std::make_shared<OctreeNode>(centerX, centerY, centerZ, size);
    }
    
    void insert(uint32_t voxelIndex, const Voxel& voxel) {
        insertRecursive(m_root, voxelIndex, voxel, 0);
    }
    
    void remove(uint32_t voxelIndex, const Voxel& voxel) {
        removeRecursive(m_root, voxelIndex, voxel);
    }
    
    void update(uint32_t voxelIndex, const Voxel& oldVoxel, const Voxel& newVoxel) {
        remove(voxelIndex, oldVoxel);
        insert(voxelIndex, newVoxel);
    }
    
    // Get all voxel indices in a given region
    std::vector<uint32_t> queryRegion(float minX, float minY, float minZ, 
                                     float maxX, float maxY, float maxZ) {
        std::vector<uint32_t> result;
        queryRegionRecursive(m_root, minX, minY, minZ, maxX, maxY, maxZ, result);
        return result;
    }

private:
    std::shared_ptr<OctreeNode> m_root;
    float m_centerX, m_centerY, m_centerZ;
    float m_size;
    uint32_t m_maxDepth;
    
    void insertRecursive(std::shared_ptr<OctreeNode> node, uint32_t voxelIndex, 
                        const Voxel& voxel, uint32_t depth) {
        // Check if this voxel is inside this node
        if (!isVoxelInNode(node, voxel.x, voxel.y, voxel.z)) {
            return;
        }
        
        // If leaf node and at max depth or has few voxels, store voxel here
        if (depth >= m_maxDepth || (node->isLeaf && node->voxelCount < 8)) {
            node->voxelIndices.push_back(voxelIndex);
            node->voxelCount++;
            node->isLeaf = true;
            return;
        }
        
        // If this is a leaf with too many voxels, subdivide
        if (node->isLeaf && node->voxelCount >= 8) {
            subdivide(node);
        }
        
        // Find appropriate child and insert there
        int octant = getOctant(node->centerX, node->centerY, node->centerZ, 
                              voxel.x, voxel.y, voxel.z);
        if (!node->children[octant]) {
            float childSize = node->size * 0.5f;
            float offsetX, offsetY, offsetZ;
            computeChildCenter(node->centerX, node->centerY, node->centerZ, 
                              node->size, octant, offsetX, offsetY, offsetZ);
            node->children[octant] = std::make_shared<OctreeNode>(
                offsetX, offsetY, offsetZ, childSize);
            node->hasChildren = true;
        }
        
        insertRecursive(node->children[octant], voxelIndex, voxel, depth + 1);
    }
    
    void removeRecursive(std::shared_ptr<OctreeNode> node, uint32_t voxelIndex, 
                        const Voxel& voxel) {
        if (!isVoxelInNode(node, voxel.x, voxel.y, voxel.z)) {
            return;
        }
        
        // Check if voxel is in this node
        auto it = std::find(node->voxelIndices.begin(), node->voxelIndices.end(), voxelIndex);
        if (it != node->voxelIndices.end()) {
            node->voxelIndices.erase(it);
            node->voxelCount--;
            return;
        }
        
        // Check children if any
        if (node->hasChildren) {
            int octant = getOctant(node->centerX, node->centerY, node->centerZ, 
                                  voxel.x, voxel.y, voxel.z);
            if (node->children[octant]) {
                removeRecursive(node->children[octant], voxelIndex, voxel);
            }
        }
    }
    
    void queryRegionRecursive(std::shared_ptr<OctreeNode> node, 
                            float minX, float minY, float minZ,
                            float maxX, float maxY, float maxZ,
                            std::vector<uint32_t>& result) {
        // Check if this node overlaps with query region
        if (!nodeOverlapsRegion(node, minX, minY, minZ, maxX, maxY, maxZ)) {
            return;
        }
        
        // Add voxels from this node
        result.insert(result.end(), node->voxelIndices.begin(), node->voxelIndices.end());
        
        // Recurse into children
        if (node->hasChildren) {
            for (auto& child : node->children) {
                if (child) {
                    queryRegionRecursive(child, minX, minY, minZ, maxX, maxY, maxZ, result);
                }
            }
        }
    }
    
    void subdivide(std::shared_ptr<OctreeNode> node) {
        // Already subdivided
        if (node->hasChildren) {
            return;
        }
        
        // Create child nodes
        for (int i = 0; i < 8; i++) {
            float childCenterX, childCenterY, childCenterZ;
            computeChildCenter(node->centerX, node->centerY, node->centerZ, 
                              node->size, i, childCenterX, childCenterY, childCenterZ);
            
            node->children[i] = std::make_shared<OctreeNode>(
                childCenterX, childCenterY, childCenterZ, node->size * 0.5f);
        }
        
        node->hasChildren = true;
        node->isLeaf = false;
    }
    
    void computeChildCenter(float parentCenterX, float parentCenterY, float parentCenterZ,
                           float parentSize, int octant,
                           float& childCenterX, float& childCenterY, float& childCenterZ) {
        float offset = parentSize * 0.25f;
        float dirX = (octant & 1) ? 1.0f : -1.0f;
        float dirY = (octant & 2) ? 1.0f : -1.0f;
        float dirZ = (octant & 4) ? 1.0f : -1.0f;
        
        childCenterX = parentCenterX + dirX * offset;
        childCenterY = parentCenterY + dirY * offset;
        childCenterZ = parentCenterZ + dirZ * offset;
    }
    
    int getOctant(float centerX, float centerY, float centerZ,
                 float posX, float posY, float posZ) {
        int octant = 0;
        if (posX >= centerX) octant |= 1;
        if (posY >= centerY) octant |= 2;
        if (posZ >= centerZ) octant |= 4;
        return octant;
    }
    
    bool isVoxelInNode(std::shared_ptr<OctreeNode> node, 
                      float posX, float posY, float posZ) {
        float halfSize = node->size * 0.5f;
        float minX = node->centerX - halfSize;
        float minY = node->centerY - halfSize;
        float minZ = node->centerZ - halfSize;
        float maxX = node->centerX + halfSize;
        float maxY = node->centerY + halfSize;
        float maxZ = node->centerZ + halfSize;
        
        return posX >= minX && posX <= maxX &&
               posY >= minY && posY <= maxY &&
               posZ >= minZ && posZ <= maxZ;
    }
    
    bool nodeOverlapsRegion(std::shared_ptr<OctreeNode> node,
                           float minX, float minY, float minZ,
                           float maxX, float maxY, float maxZ) {
        float halfSize = node->size * 0.5f;
        float nodeMinX = node->centerX - halfSize;
        float nodeMinY = node->centerY - halfSize;
        float nodeMinZ = node->centerZ - halfSize;
        float nodeMaxX = node->centerX + halfSize;
        float nodeMaxY = node->centerY + halfSize;
        float nodeMaxZ = node->centerZ + halfSize;
        
        return nodeMinX <= maxX && nodeMaxX >= minX &&
               nodeMinY <= maxY && nodeMaxY >= minY &&
               nodeMinZ <= maxZ && nodeMaxZ >= minZ;
    }
};

// GPU Voxel Manager - Handles uploading and updating voxel data on GPU
class GPUVoxelManager {
public:
    GPUVoxelManager(WGPUDevice device, uint32_t maxVoxels) 
        : m_device(device), m_maxVoxels(maxVoxels) {
        initializeBuffers();
    }
    
    ~GPUVoxelManager() {
        // Clean up WebGPU resources
        if (m_voxelBuffer) {
            wgpuBufferDestroy(m_voxelBuffer);
            m_voxelBuffer = nullptr;
        }
        if (m_visibleIndicesBuffer) {
            wgpuBufferDestroy(m_visibleIndicesBuffer);
            m_visibleIndicesBuffer = nullptr;
        }
        if (m_visibleCountBuffer) {
            wgpuBufferDestroy(m_visibleCountBuffer);
            m_visibleCountBuffer = nullptr;
        }
    }
    
    void updateVoxelData(const std::vector<Voxel>& voxels) {
        // Only update if we have data to update
        if (voxels.empty()) {
            return;
        }
        
        size_t dataSize = voxels.size() * sizeof(Voxel);
        
        // Check if we need to resize
        if (dataSize > m_voxelBufferInfo.capacity) {
            resizeVoxelBuffer(dataSize);
        }
        
        // Update the buffer
        wgpuQueueWriteBuffer(queue, m_voxelBuffer, 0, voxels.data(), dataSize);
        
        m_voxelBufferInfo.size = dataSize;
        m_voxelCount = static_cast<uint32_t>(voxels.size());
        
        // Reset visible count
        uint32_t zero = 0;
        wgpuQueueWriteBuffer(queue, m_visibleCountBuffer, 0, &zero, sizeof(uint32_t));

        // Update global references
        voxelBuffer = m_voxelBuffer;
        visibleIndicesBuffer = m_visibleIndicesBuffer;
        visibleCountBuffer = m_visibleCountBuffer;
    }
    
    void updateVoxelRange(const std::vector<Voxel>& voxels, uint32_t startIdx, uint32_t count) {
        // Calculate offset and ensure we're within bounds
        size_t offset = startIdx * sizeof(Voxel);
        size_t dataSize = count * sizeof(Voxel);
        
        if (startIdx + count > m_maxVoxels) {
            LogMessage("Error: Voxel range out of bounds");
            return;
        }
        
        // Update the buffer at the specified range
        wgpuQueueWriteBuffer(queue, m_voxelBuffer, offset, &voxels[startIdx], dataSize);
    }
    
    uint32_t getVoxelCount() const { return m_voxelCount; }
    WGPUBuffer getVoxelBuffer() const { return m_voxelBuffer; }
    WGPUBuffer getVisibleIndicesBuffer() const { return m_visibleIndicesBuffer; }
    WGPUBuffer getVisibleCountBuffer() const { return m_visibleCountBuffer; }

private:
    WGPUDevice m_device;
    uint32_t m_maxVoxels;
    uint32_t m_voxelCount = 0;
    
    WGPUBuffer m_voxelBuffer = nullptr;
    WGPUBuffer m_visibleIndicesBuffer = nullptr;
    WGPUBuffer m_visibleCountBuffer = nullptr;
    
    GPUBufferInfo m_voxelBufferInfo{};
    GPUBufferInfo m_visibleIndicesBufferInfo{};
    GPUBufferInfo m_visibleCountBufferInfo{};
    
    void initializeBuffers() {
        // Initial capacity for voxel data
        size_t initialCapacity = MAX_VOXELS * sizeof(Voxel); // Changed from 1024 to MAX_VOXELS
        
        // Create voxel storage buffer
        WGPUBufferDescriptor voxelBufferDesc{};
        voxelBufferDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        voxelBufferDesc.size = initialCapacity;
        voxelBufferDesc.mappedAtCreation = false;
        
        m_voxelBuffer = wgpuDeviceCreateBuffer(m_device, &voxelBufferDesc);
        m_voxelBufferInfo.buffer = m_voxelBuffer;
        m_voxelBufferInfo.capacity = initialCapacity;
        m_voxelBufferInfo.size = 0;
        m_voxelBufferInfo.needsUpdate = false;
        
        // Create visible voxel indices buffer (used for indirect drawing)
        WGPUBufferDescriptor visibleIndicesBufferDesc{};
        visibleIndicesBufferDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
        visibleIndicesBufferDesc.size = m_maxVoxels * sizeof(uint32_t);
        visibleIndicesBufferDesc.mappedAtCreation = false;
        
        m_visibleIndicesBuffer = wgpuDeviceCreateBuffer(m_device, &visibleIndicesBufferDesc);
        m_visibleIndicesBufferInfo.buffer = m_visibleIndicesBuffer;
        m_visibleIndicesBufferInfo.capacity = m_maxVoxels * sizeof(uint32_t);
        m_visibleIndicesBufferInfo.size = m_maxVoxels * sizeof(uint32_t);
        m_visibleIndicesBufferInfo.needsUpdate = false;
        
        // Create visible voxel count buffer (used for indirect drawing)
        WGPUBufferDescriptor visibleCountBufferDesc{};
        visibleCountBufferDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
        visibleCountBufferDesc.size = sizeof(uint32_t);
        visibleCountBufferDesc.mappedAtCreation = false;
        
        m_visibleCountBuffer = wgpuDeviceCreateBuffer(m_device, &visibleCountBufferDesc);
        m_visibleCountBufferInfo.buffer = m_visibleCountBuffer;
        m_visibleCountBufferInfo.capacity = sizeof(uint32_t);
        m_visibleCountBufferInfo.size = sizeof(uint32_t);
        m_visibleCountBufferInfo.needsUpdate = false;
        
        // Initialize visible count to 0
        uint32_t zero = 0;
        wgpuQueueWriteBuffer(queue, m_visibleCountBuffer, 0, &zero, sizeof(uint32_t));

        // Set global references
        voxelBuffer = m_voxelBuffer;
        visibleIndicesBuffer = m_visibleIndicesBuffer;
        visibleCountBuffer = m_visibleCountBuffer;
    }
    
    void resizeVoxelBuffer(size_t requiredSize) {
        // Calculate new capacity with some growth factor
        size_t newCapacity = m_voxelBufferInfo.capacity;
        while (newCapacity < requiredSize) {
            newCapacity *= 2;
        }
        
        // Create new buffer
        WGPUBufferDescriptor newBufferDesc{};
        newBufferDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        newBufferDesc.size = newCapacity;
        newBufferDesc.mappedAtCreation = false;
        
        WGPUBuffer newBuffer = wgpuDeviceCreateBuffer(m_device, &newBufferDesc);
        
        // Copy data from old buffer if needed
        if (m_voxelBufferInfo.size > 0) {
            // Create command encoder
            WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(m_device, nullptr);
            
            // Copy from old to new
            wgpuCommandEncoderCopyBufferToBuffer(
                encoder,
                m_voxelBuffer, 0,
                newBuffer, 0,
                m_voxelBufferInfo.size
            );
            
            // Finish and submit commands
            WGPUCommandBufferDescriptor cmdBufferDesc = {};
            WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &cmdBufferDesc);
            wgpuQueueSubmit(queue, 1, &commands);
            
            // Cleanup
            wgpuCommandBufferRelease(commands);
            wgpuCommandEncoderRelease(encoder);
        }
        
        // Clean up old buffer
        wgpuBufferDestroy(m_voxelBuffer);
        
        // Update with new buffer
        m_voxelBuffer = newBuffer;
        m_voxelBufferInfo.buffer = newBuffer;
        m_voxelBufferInfo.capacity = newCapacity;
        
        // Update global reference
        voxelBuffer = m_voxelBuffer;
    }
};

// Voxel System - Manages the voxel data and octree
class VoxelSystem {
public:
    VoxelSystem(uint32_t maxVoxels, float worldCenterX, float worldCenterY, float worldCenterZ, float worldSize)
        : m_maxVoxels(maxVoxels), 
          m_octree(std::make_unique<VoxelOctree>(worldCenterX, worldCenterY, worldCenterZ, worldSize, OCTREE_MAX_DEPTH)) {
        m_voxels.reserve(maxVoxels);
    }
    
    uint32_t addVoxel(const Voxel& voxel) {
        if (m_voxels.size() >= m_maxVoxels) {
            LogMessage("Warning: Maximum voxel count reached");
            return UINT32_MAX;
        }
        
        uint32_t index = static_cast<uint32_t>(m_voxels.size());
        m_voxels.push_back(voxel);
        
        // Add to octree
        m_octree->insert(index, voxel);
        
        // Mark as dirty for GPU update
        m_isDirty = true;
        
        return index;
    }
    
    void removeVoxel(uint32_t index) {
        if (index >= m_voxels.size()) {
            return;
        }
        
        // Mark as inactive instead of removing (to maintain indices)
        m_voxels[index].isActive = false;
        
        // Mark as dirty for GPU update
        m_isDirty = true;
    }
    
    void updateVoxel(uint32_t index, const Voxel& newVoxel) {
        if (index >= m_voxels.size()) {
            return;
        }
        
        Voxel oldVoxel = m_voxels[index];
        m_voxels[index] = newVoxel;
        
        // Update in octree if position changed
        if (oldVoxel.x != newVoxel.x || oldVoxel.y != newVoxel.y || oldVoxel.z != newVoxel.z) {
            m_octree->update(index, oldVoxel, newVoxel);
        }
        
        // Mark as dirty for GPU update
        m_isDirty = true;
    }
    
    // Get voxels in region
    std::vector<uint32_t> getVoxelsInRegion(float minX, float minY, float minZ, 
                                           float maxX, float maxY, float maxZ) {
        return m_octree->queryRegion(minX, minY, minZ, maxX, maxY, maxZ);
    }
    
    // Apply function to each voxel
    void forEachVoxel(const std::function<void(uint32_t, Voxel&)>& func) {
        for (uint32_t i = 0; i < m_voxels.size(); ++i) {
            if (m_voxels[i].isActive) {
                func(i, m_voxels[i]);
            }
        }
        
        // Mark as dirty for GPU update
        m_isDirty = true;
    }
    
    void uploadToGPU(GPUVoxelManager& gpuManager) {
        if (!m_isDirty) {
            return;
        }
        
        // Upload all voxels to GPU
        gpuManager.updateVoxelData(m_voxels);
        
        m_isDirty = false;
    }
    
    const std::vector<Voxel>& getVoxels() const { return m_voxels; }
    bool isDirty() const { return m_isDirty; }

private:
    uint32_t m_maxVoxels;
    std::vector<Voxel> m_voxels;
    std::unique_ptr<VoxelOctree> m_octree;
    bool m_isDirty = false;
};

// Shader sources
// Vertex shader for voxel rendering
static const char* voxelVertexShaderSource = R"(
// Voxel structure
struct Voxel {
    position: vec3f,
    color: u32,
    voxelType: u32,   // Changed from 'type' to 'voxelType'
    isActive: u32,    // Changed from 'active' to 'isActive'
}

// Camera data
struct CameraData {
    viewProjection: mat4x4f,
    cameraPosition: vec3f,
    padding: f32,
    frustumPlanes: array<vec4f, 6>,
    maxDrawDistance: f32,
}

// Bindings
@group(0) @binding(0) var<uniform> camera: CameraData;
@group(0) @binding(1) var<storage, read> voxels: array<Voxel>;

// Cube vertices for instancing
const positions = array<vec3f, 8>(
    vec3f(-0.5, -0.5, -0.5),
    vec3f( 0.5, -0.5, -0.5),
    vec3f(-0.5,  0.5, -0.5),
    vec3f( 0.5,  0.5, -0.5),
    vec3f(-0.5, -0.5,  0.5),
    vec3f( 0.5, -0.5,  0.5),
    vec3f(-0.5,  0.5,  0.5),
    vec3f( 0.5,  0.5,  0.5)
);

// Cube indices (36 vertices = 12 triangles)
const indices = array<u32, 36>(
    0, 2, 1, 1, 2, 3, // front
    4, 5, 6, 5, 7, 6, // back
    0, 4, 2, 2, 4, 6, // left
    1, 3, 5, 3, 7, 5, // right
    2, 6, 3, 3, 6, 7, // top
    0, 1, 4, 1, 5, 4  // bottom
);

// Convert uint color to vec4 float color
fn unpackColor(color: u32) -> vec4f {
    return vec4f(
        f32((color >> 24) & 0xFF) / 255.0,
        f32((color >> 16) & 0xFF) / 255.0,
        f32((color >> 8) & 0xFF) / 255.0,
        f32(color & 0xFF) / 255.0
    );
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) worldPos: vec3f,
}

@vertex
fn main(@builtin(instance_index) instanceIdx: u32, 
        @builtin(vertex_index) vertexIdx: u32) -> VertexOutput {
    // Get the voxel for this instance
    let voxel = voxels[instanceIdx];
    
    // Skip inactive voxels
    if (voxel.isActive == 0u) {   // Changed from 'active' to 'isActive'
        return VertexOutput(vec4f(0.0, 0.0, 0.0, 0.0), vec4f(0.0), vec3f(0.0));
    }
    
    // Get vertex position from cube vertices using indexed geometry
    let idx = indices[vertexIdx];
    let vertexPos = positions[idx];
    
    // Calculate world space position
    let worldPos = vertexPos + voxel.position;
    
    // Calculate clip space position
    let clipPos = camera.viewProjection * vec4f(worldPos, 1.0);
    
    // Unpack color
    let color = unpackColor(voxel.color);
    
    return VertexOutput(clipPos, color, worldPos);
}
)";

// Fragment shader for voxel rendering
static const char* voxelFragmentShaderSource = R"(
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) worldPos: vec3f,
}

@fragment
fn main(in: VertexOutput) -> @location(0) vec4f {
    return in.color;
}
)";

// Compute shader for voxel culling
static const char* voxelComputeShaderSource = R"(
// Camera data
struct CameraData {
    viewProjection: mat4x4f,
    cameraPosition: vec3f,
    padding: f32,
    frustumPlanes: array<vec4f, 6>,
    maxDrawDistance: f32,
}

// Voxel structure
struct Voxel {
    position: vec3f,
    color: u32,
    voxelType: u32,   // Changed from 'type' to 'voxelType'
    isActive: u32,    // Changed from 'active' to 'isActive'
}

// Bindings
@group(0) @binding(0) var<uniform> camera: CameraData;
@group(0) @binding(1) var<storage, read> voxels: array<Voxel>;
@group(0) @binding(2) var<storage, read_write> visibleVoxelIndices: array<u32>;
@group(0) @binding(3) var<storage, read_write> visibleVoxelCount: atomic<u32>;

// Check if a voxel is inside view frustum
fn isVoxelVisible(position: vec3f, size: f32) -> bool {
    // Distance culling
    let distToCam = distance(position, camera.cameraPosition);
    if (distToCam > camera.maxDrawDistance) {
        return false;
    }
    
    // Frustum culling
    // Create an AABB for the voxel
    let halfSize = size * 0.5;
    let minPos = position - vec3f(halfSize);
    let maxPos = position + vec3f(halfSize);
    
    // Check against each frustum plane
    for (var i = 0u; i < 6u; i++) {
        let plane = camera.frustumPlanes[i];
        let normal = plane.xyz;
        let constant = plane.w;
        
        // Find the positive vertex
        let p = select(minPos, maxPos, normal > vec3f(0.0));
        
        // If this point is outside the plane, the whole box is outside
        if (dot(normal, p) + constant < 0.0) {
            return false;
        }
    }
    
    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&voxels)) {
        return;
    }
    
    var voxel = voxels[index];
    
    // Skip processing if voxel is not active
    if (voxel.isActive == 0u) {   // Changed from 'active' to 'isActive'
        return;
    }
    
    // Check if voxel is visible
    let isVisible = isVoxelVisible(voxel.position, 1.0); // Assuming voxel size of 1.0
    
    if (isVisible) {
        // Add to visible voxels list
        let idx = atomicAdd(&visibleVoxelCount, 1u);
        if (idx < arrayLength(&visibleVoxelIndices)) {
            visibleVoxelIndices[idx] = index;
        }
    }
}
)";

// Function to extract frustum planes from view-projection matrix
void ExtractFrustumPlanes(const float viewProj[16], float planes[6][4]) {
    // Left plane
    planes[0][0] = viewProj[3] + viewProj[0];
    planes[0][1] = viewProj[7] + viewProj[4];
    planes[0][2] = viewProj[11] + viewProj[8];
    planes[0][3] = viewProj[15] + viewProj[12];
    
    // Right plane
    planes[1][0] = viewProj[3] - viewProj[0];
    planes[1][1] = viewProj[7] - viewProj[4];
    planes[1][2] = viewProj[11] - viewProj[8];
    planes[1][3] = viewProj[15] - viewProj[12];
    
    // Bottom plane
    planes[2][0] = viewProj[3] + viewProj[1];
    planes[2][1] = viewProj[7] + viewProj[5];
    planes[2][2] = viewProj[11] + viewProj[9];
    planes[2][3] = viewProj[15] + viewProj[13];
    
    // Top plane
    planes[3][0] = viewProj[3] - viewProj[1];
    planes[3][1] = viewProj[7] - viewProj[5];
    planes[3][2] = viewProj[11] - viewProj[9];
    planes[3][3] = viewProj[15] - viewProj[13];
    
    // Near plane
    planes[4][0] = viewProj[3] + viewProj[2];
    planes[4][1] = viewProj[7] + viewProj[6];
    planes[4][2] = viewProj[11] + viewProj[10];
    planes[4][3] = viewProj[15] + viewProj[14];
    
    // Far plane
    planes[5][0] = viewProj[3] - viewProj[2];
    planes[5][1] = viewProj[7] - viewProj[6];
    planes[5][2] = viewProj[11] - viewProj[10];
    planes[5][3] = viewProj[15] - viewProj[14];
    
    // Normalize all planes
    for (int i = 0; i < 6; ++i) {
        float length = sqrt(planes[i][0] * planes[i][0] + 
                           planes[i][1] * planes[i][1] +
                           planes[i][2] * planes[i][2]);
        planes[i][0] /= length;
        planes[i][1] /= length;
        planes[i][2] /= length;
        planes[i][3] /= length;
    }
}

// Camera state
struct Camera {
    float position[3] = {0.0f, 0.0f, -15.0f};
    float target[3] = {0.0f, 0.0f, 0.0f};
    float up[3] = {0.0f, 1.0f, 0.0f};
    float rotation = 0.0f;
    float viewMatrix[16];
    float projMatrix[16];
    float viewProjMatrix[16];
    float aspectRatio = (float)CANVAS_WIDTH / CANVAS_HEIGHT;
    float fov = 60.0f;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;
    CameraData gpuData;
};

static Camera camera;

// Matrix utility functions
void Identity(float* matrix) {
    matrix[0] = 1.0f; matrix[4] = 0.0f; matrix[8] = 0.0f; matrix[12] = 0.0f;
    matrix[1] = 0.0f; matrix[5] = 1.0f; matrix[9] = 0.0f; matrix[13] = 0.0f;
    matrix[2] = 0.0f; matrix[6] = 0.0f; matrix[10] = 1.0f; matrix[14] = 0.0f;
    matrix[3] = 0.0f; matrix[7] = 0.0f; matrix[11] = 0.0f; matrix[15] = 1.0f;
}

void LookAt(float* matrix, const float* eye, const float* target, const float* up) {
    float f[3];
    f[0] = target[0] - eye[0];
    f[1] = target[1] - eye[1];
    f[2] = target[2] - eye[2];
    
    // Normalize f
    float len = sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    f[0] /= len;
    f[1] /= len;
    f[2] /= len;
    
    // Compute s = f x up
    float s[3];
    s[0] = f[1] * up[2] - f[2] * up[1];
    s[1] = f[2] * up[0] - f[0] * up[2];
    s[2] = f[0] * up[1] - f[1] * up[0];
    
    // Normalize s
    len = sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    s[0] /= len;
    s[1] /= len;
    s[2] /= len;
    
    // Compute u = s x f
    float u[3];
    u[0] = s[1] * f[2] - s[2] * f[1];
    u[1] = s[2] * f[0] - s[0] * f[2];
    u[2] = s[0] * f[1] - s[1] * f[0];
    
    // Build matrix
    matrix[0] = s[0]; matrix[4] = s[1]; matrix[8] = s[2]; matrix[12] = -s[0] * eye[0] - s[1] * eye[1] - s[2] * eye[2];
    matrix[1] = u[0]; matrix[5] = u[1]; matrix[9] = u[2]; matrix[13] = -u[0] * eye[0] - u[1] * eye[1] - u[2] * eye[2];
    matrix[2] = -f[0]; matrix[6] = -f[1]; matrix[10] = -f[2]; matrix[14] = f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2];
    matrix[3] = 0.0f; matrix[7] = 0.0f; matrix[11] = 0.0f; matrix[15] = 1.0f;
}

void Perspective(float* matrix, float fovY, float aspect, float near, float far) {
    float f = 1.0f / tan(fovY * (3.14159f / 180.0f) * 0.5f);
    
    matrix[0] = f / aspect; matrix[4] = 0.0f; matrix[8] = 0.0f; matrix[12] = 0.0f;
    matrix[1] = 0.0f; matrix[5] = f; matrix[9] = 0.0f; matrix[13] = 0.0f;
    matrix[2] = 0.0f; matrix[6] = 0.0f; matrix[10] = (far + near) / (near - far); matrix[14] = (2.0f * far * near) / (near - far);
    matrix[3] = 0.0f; matrix[7] = 0.0f; matrix[11] = -1.0f; matrix[15] = 0.0f;
}

void MultiplyMatrix(const float* a, const float* b, float* result) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result[i * 4 + j] = 
                a[i * 4 + 0] * b[0 * 4 + j] +
                a[i * 4 + 1] * b[1 * 4 + j] +
                a[i * 4 + 2] * b[2 * 4 + j] +
                a[i * 4 + 3] * b[3 * 4 + j];
        }
    }
}

// Update camera matrices
void UpdateCamera() {
    // Update camera position based on time
    float time = static_cast<float>(emscripten_get_now()) / 1000.0f;
    camera.rotation = time * 0.5f;
    float radius = 15.0f;
    
    camera.position[0] = sin(camera.rotation) * radius;
    camera.position[2] = cos(camera.rotation) * radius;
    camera.position[1] = 5.0f + sin(time * 0.3f) * 2.0f;
    
    // Create view matrix
    LookAt(camera.viewMatrix, camera.position, camera.target, camera.up);
    
    // Create projection matrix
    Perspective(camera.projMatrix, camera.fov, camera.aspectRatio, camera.nearPlane, camera.farPlane);
    
    // Combine view and projection
    MultiplyMatrix(camera.projMatrix, camera.viewMatrix, camera.viewProjMatrix);
    
    // Update camera data for GPU
    for (int i = 0; i < 16; ++i) {
        camera.gpuData.viewProjection[i] = camera.viewProjMatrix[i];
    }
    
    camera.gpuData.cameraPosition[0] = camera.position[0];
    camera.gpuData.cameraPosition[1] = camera.position[1];
    camera.gpuData.cameraPosition[2] = camera.position[2];
    camera.gpuData.padding = 0.0f;
    camera.gpuData.maxDrawDistance = 100.0f;
    
    // Calculate frustum planes
    ExtractFrustumPlanes(camera.viewProjMatrix, camera.gpuData.frustumPlanes);
}

// Upload camera data to GPU
void UploadCameraData() {
    wgpuQueueWriteBuffer(queue, cameraBuffer, 0, &camera.gpuData, sizeof(CameraData));
}

// Create rendering pipelines
bool CreateRenderPipelines() {
    LogMessage("Creating render pipelines...");
    
    // Create camera uniform buffer
    WGPUBufferDescriptor cameraBufferDesc{};
    cameraBufferDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    cameraBufferDesc.size = 192; // Update to 192 bytes instead of sizeof(CameraData)
    cameraBufferDesc.mappedAtCreation = false;
    
    cameraBuffer = wgpuDeviceCreateBuffer(device, &cameraBufferDesc);
    if (!cameraBuffer) {
        LogMessage("Failed to create camera buffer");
        return false;
    }
    
    // Create depth texture
    WGPUTextureDescriptor depthTextureDesc{};
    depthTextureDesc.dimension = WGPUTextureDimension_2D;
    depthTextureDesc.size = { CANVAS_WIDTH, CANVAS_HEIGHT, 1 };
    depthTextureDesc.format = WGPUTextureFormat_Depth24Plus;
    depthTextureDesc.mipLevelCount = 1;
    depthTextureDesc.sampleCount = 1;
    depthTextureDesc.usage = WGPUTextureUsage_RenderAttachment;
    
    depthTexture = wgpuDeviceCreateTexture(device, &depthTextureDesc);
    if (!depthTexture) {
        LogMessage("Failed to create depth texture");
        return false;
    }
    
    WGPUTextureViewDescriptor depthTextureViewDesc{};
    depthTextureViewDesc.format = WGPUTextureFormat_Depth24Plus;
    depthTextureViewDesc.dimension = WGPUTextureViewDimension_2D;
    depthTextureViewDesc.baseMipLevel = 0;
    depthTextureViewDesc.mipLevelCount = 1;
    depthTextureViewDesc.baseArrayLayer = 0;
    depthTextureViewDesc.arrayLayerCount = 1;
    depthTextureViewDesc.aspect = WGPUTextureAspect_DepthOnly;
    
    depthTextureView = wgpuTextureCreateView(depthTexture, &depthTextureViewDesc);
    if (!depthTextureView) {
        LogMessage("Failed to create depth texture view");
        return false;
    }
    
    // Create shader modules
    WGPUShaderModuleDescriptor vertexShaderDesc{};
    WGPUShaderModuleWGSLDescriptor vertexShaderCodeDesc{};
    vertexShaderCodeDesc.code = voxelVertexShaderSource;
    vertexShaderCodeDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    vertexShaderDesc.nextInChain = &vertexShaderCodeDesc.chain;
    
    WGPUShaderModule vertexShaderModule = wgpuDeviceCreateShaderModule(device, &vertexShaderDesc);
    if (!vertexShaderModule) {
        LogMessage("Failed to create vertex shader module");
        return false;
    }
    
    WGPUShaderModuleDescriptor fragmentShaderDesc{};
    WGPUShaderModuleWGSLDescriptor fragmentShaderCodeDesc{};
    fragmentShaderCodeDesc.code = voxelFragmentShaderSource;
    fragmentShaderCodeDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    fragmentShaderDesc.nextInChain = &fragmentShaderCodeDesc.chain;
    
    WGPUShaderModule fragmentShaderModule = wgpuDeviceCreateShaderModule(device, &fragmentShaderDesc);
    if (!fragmentShaderModule) {
        LogMessage("Failed to create fragment shader module");
        wgpuShaderModuleRelease(vertexShaderModule);
        return false;
    }
    
    WGPUShaderModuleDescriptor computeShaderDesc{};
    WGPUShaderModuleWGSLDescriptor computeShaderCodeDesc{};
    computeShaderCodeDesc.code = voxelComputeShaderSource;
    computeShaderCodeDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    computeShaderDesc.nextInChain = &computeShaderCodeDesc.chain;
    
    WGPUShaderModule computeShaderModule = wgpuDeviceCreateShaderModule(device, &computeShaderDesc);
    if (!computeShaderModule) {
        LogMessage("Failed to create compute shader module");
        wgpuShaderModuleRelease(vertexShaderModule);
        wgpuShaderModuleRelease(fragmentShaderModule);
        return false;
    }
    
    // Create bind group layout
    WGPUBindGroupLayoutEntry bindGroupLayoutEntries[4] = {};
    
    // Camera uniform buffer
    bindGroupLayoutEntries[0].binding = 0;
    bindGroupLayoutEntries[0].visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Compute;
    bindGroupLayoutEntries[0].buffer.type = WGPUBufferBindingType_Uniform;
    bindGroupLayoutEntries[0].buffer.minBindingSize = 192; // Changed from sizeof(CameraData) to 192 to match shader
    
    // Voxel storage buffer - changed to read-only for vertex shader
    bindGroupLayoutEntries[1].binding = 1;
    bindGroupLayoutEntries[1].visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Compute;
    bindGroupLayoutEntries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    bindGroupLayoutEntries[1].buffer.minBindingSize = 32; // Update to 32 bytes to match shader's expectation
    
    // Visible indices storage buffer
    bindGroupLayoutEntries[2].binding = 2;
    bindGroupLayoutEntries[2].visibility = WGPUShaderStage_Compute;
    bindGroupLayoutEntries[2].buffer.type = WGPUBufferBindingType_Storage;
    bindGroupLayoutEntries[2].buffer.minBindingSize = sizeof(uint32_t);
    
    // Visible count atomic buffer
    bindGroupLayoutEntries[3].binding = 3;
    bindGroupLayoutEntries[3].visibility = WGPUShaderStage_Compute;
    bindGroupLayoutEntries[3].buffer.type = WGPUBufferBindingType_Storage;
    bindGroupLayoutEntries[3].buffer.minBindingSize = sizeof(uint32_t);
    
    WGPUBindGroupLayoutDescriptor bindGroupLayoutDesc{};
    bindGroupLayoutDesc.entryCount = 4;
    bindGroupLayoutDesc.entries = bindGroupLayoutEntries;
    
    bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bindGroupLayoutDesc);
    if (!bindGroupLayout) {
        LogMessage("Failed to create bind group layout");
        wgpuShaderModuleRelease(vertexShaderModule);
        wgpuShaderModuleRelease(fragmentShaderModule);
        wgpuShaderModuleRelease(computeShaderModule);
        return false;
    }
    
    // Initialize voxel manager and create initial voxel buffer
    gpuVoxelManager = std::make_unique<GPUVoxelManager>(device, MAX_VOXELS);
    
    // Create and return pipeline layout only, we'll create bind groups later
    WGPUPipelineLayoutDescriptor pipelineLayoutDesc{};
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &bindGroupLayout;
    
    WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &pipelineLayoutDesc);
    if (!pipelineLayout) {
        LogMessage("Failed to create pipeline layout");
        wgpuShaderModuleRelease(vertexShaderModule);
        wgpuShaderModuleRelease(fragmentShaderModule);
        wgpuShaderModuleRelease(computeShaderModule);
        return false;
    }
    
    // Create compute pipeline
    WGPUComputePipelineDescriptor computePipelineDesc{};
    computePipelineDesc.layout = pipelineLayout;
    computePipelineDesc.compute.module = computeShaderModule;
    computePipelineDesc.compute.entryPoint = "main";
    
    computePipeline = wgpuDeviceCreateComputePipeline(device, &computePipelineDesc);
    if (!computePipeline) {
        LogMessage("Failed to create compute pipeline");
        wgpuPipelineLayoutRelease(pipelineLayout);
        wgpuShaderModuleRelease(vertexShaderModule);
        wgpuShaderModuleRelease(fragmentShaderModule);
        wgpuShaderModuleRelease(computeShaderModule);
        return false;
    }
    
    // Create render pipeline
    WGPUBlendState blendState{};
    blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
    blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blendState.color.operation = WGPUBlendOperation_Add;
    blendState.alpha.srcFactor = WGPUBlendFactor_One;
    blendState.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blendState.alpha.operation = WGPUBlendOperation_Add;
    
    WGPUColorTargetState colorTarget{};
    colorTarget.format = WGPUTextureFormat_BGRA8Unorm;
    colorTarget.blend = &blendState;
    colorTarget.writeMask = WGPUColorWriteMask_All;
    
    WGPUFragmentState fragmentState{};
    fragmentState.module = fragmentShaderModule;
    fragmentState.entryPoint = "main";
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTarget;
    
    WGPUDepthStencilState depthStencilState{};
    depthStencilState.format = WGPUTextureFormat_Depth24Plus;
    depthStencilState.depthWriteEnabled = true;
    depthStencilState.depthCompare = WGPUCompareFunction_Less;
    
    WGPURenderPipelineDescriptor renderPipelineDesc{};
    renderPipelineDesc.layout = pipelineLayout;
    renderPipelineDesc.vertex.module = vertexShaderModule;
    renderPipelineDesc.vertex.entryPoint = "main";
    renderPipelineDesc.vertex.bufferCount = 0; // No vertex buffers, we use instancing
    renderPipelineDesc.vertex.constantCount = 0;
    renderPipelineDesc.vertex.constants = nullptr;
    renderPipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    renderPipelineDesc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    renderPipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
    renderPipelineDesc.primitive.cullMode = WGPUCullMode_Back;
    renderPipelineDesc.depthStencil = &depthStencilState;
    renderPipelineDesc.multisample.count = 1;
    renderPipelineDesc.multisample.mask = 0xFFFFFFFF;
    renderPipelineDesc.multisample.alphaToCoverageEnabled = false;
    renderPipelineDesc.fragment = &fragmentState;
    
    renderPipeline = wgpuDeviceCreateRenderPipeline(device, &renderPipelineDesc);
    if (!renderPipeline) {
        LogMessage("Failed to create render pipeline");
        wgpuComputePipelineRelease(computePipeline);
        wgpuPipelineLayoutRelease(pipelineLayout);
        wgpuShaderModuleRelease(vertexShaderModule);
        wgpuShaderModuleRelease(fragmentShaderModule);
        wgpuShaderModuleRelease(computeShaderModule);
        return false;
    }
    
    // Clean up
    wgpuPipelineLayoutRelease(pipelineLayout);
    wgpuShaderModuleRelease(vertexShaderModule);
    wgpuShaderModuleRelease(fragmentShaderModule);
    wgpuShaderModuleRelease(computeShaderModule);
    
    LogMessage("Render pipelines created successfully");
    return true;
}

// Create bind groups after voxels are created
void CreateBindGroups() {
    // Skip if already created or not ready
    if (renderBindGroup != nullptr || bindGroupLayout == nullptr || voxelBuffer == nullptr) {
        return;
    }
    
    // Create bind groups (after creating voxels)
    WGPUBindGroupEntry bindGroupEntries[4] = {};
    
    // Camera uniform buffer
    bindGroupEntries[0].binding = 0;
    bindGroupEntries[0].buffer = cameraBuffer;
    bindGroupEntries[0].offset = 0;
    bindGroupEntries[0].size = 192; // Update to 192 bytes
    
    // Voxel storage buffer
    bindGroupEntries[1].binding = 1;
    bindGroupEntries[1].buffer = voxelBuffer;
    bindGroupEntries[1].offset = 0;
    // Use minimum size of 32 bytes (one voxel) if no voxels yet
    uint32_t voxelCount = gpuVoxelManager->getVoxelCount();
    bindGroupEntries[1].size = voxelCount > 0 ? voxelCount * sizeof(Voxel) : 32;
    
    // Visible indices storage buffer
    bindGroupEntries[2].binding = 2;
    bindGroupEntries[2].buffer = visibleIndicesBuffer;
    bindGroupEntries[2].offset = 0;
    bindGroupEntries[2].size = sizeof(uint32_t) * MAX_VOXELS;
    
    // Visible count atomic buffer
    bindGroupEntries[3].binding = 3;
    bindGroupEntries[3].buffer = visibleCountBuffer;
    bindGroupEntries[3].offset = 0;
    bindGroupEntries[3].size = sizeof(uint32_t);
    
    WGPUBindGroupDescriptor bindGroupDesc{};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 4;
    bindGroupDesc.entries = bindGroupEntries;
    
    renderBindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
    computeBindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
    
    if (!renderBindGroup || !computeBindGroup) {
        LogMessage("Failed to create bind groups");
        hasError = true;
    } else {
        LogMessage("Bind groups created successfully");
    }
}

// Create example voxels
void CreateExampleVoxels() {
    std::default_random_engine generator(static_cast<unsigned int>(emscripten_get_now()));
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    std::uniform_int_distribution<int> colorDist(0, 255);
    std::uniform_int_distribution<int> typeDist(0, 2);
    
    // Pre-allocate a vector to hold all potential voxels
    std::vector<Voxel> newVoxels;
    newVoxels.reserve(1000); // Reserve space for up to 1000 voxels
    
    for (int x = -5; x <= 5; ++x) {
        for (int y = -5; y <= 5; ++y) {
            for (int z = -5; z <= 5; ++z) {
                // Skip creating some voxels to make it less dense
                if (generator() % 2 == 0) {
                    continue;
                }
                
                // Add some random offset to position
                float offsetX = distribution(generator);
                float offsetY = distribution(generator);
                float offsetZ = distribution(generator);
                
                Voxel voxel;
                voxel.x = static_cast<float>(x) + offsetX;
                voxel.y = static_cast<float>(y) + offsetY;
                voxel.z = static_cast<float>(z) + offsetZ;
                
                // Random color
                uint8_t r = colorDist(generator);
                uint8_t g = colorDist(generator);
                uint8_t b = colorDist(generator);
                uint8_t a = 255;
                voxel.color = (r << 24) | (g << 16) | (b << 8) | a;
                
                // Random type
                voxel.voxelType = typeDist(generator);
                voxel.isActive = true;
                
                voxelSystem->addVoxel(voxel);
            }
        }
    }
}

// Update voxels animation
void UpdateVoxels() {
    float time = static_cast<float>(emscripten_get_now()) / 1000.0f;
    
    voxelSystem->forEachVoxel([time](uint32_t index, Voxel& voxel) {
        // Different behavior based on voxel type
        switch (voxel.voxelType) {
            case 0: // Static voxels
                break;
                
            case 1: // Floating voxels
                voxel.y += sin(time * 0.5f + index * 0.1f) * 0.01f;
                break;
                
            case 2: { // Orbiting voxels
                float angle = time * 0.2f + index * 0.01f;
                float radius = 5.0f + sin(index * 0.1f) * 2.0f;
                
                voxel.x = sin(angle) * radius;
                voxel.z = cos(angle) * radius;
                break;
            }
        }
    });
}

// Draw a single frame
void DrawFrame() {
    if (!initialized || !device || !swapChain || !renderPipeline || hasError) {
        return;
    }

    // Update camera
    UpdateCamera();
    UploadCameraData();
    
    // Update voxels
    UpdateVoxels();
    
    // Upload voxel data if dirty
    if (voxelSystem->isDirty()) {
        voxelSystem->uploadToGPU(*gpuVoxelManager);
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
    
    // Reset visible voxel count
    uint32_t zero = 0;
    wgpuQueueWriteBuffer(queue, visibleCountBuffer, 0, &zero, sizeof(uint32_t));
    
    // First, run compute pass for culling
    {
        WGPUComputePassDescriptor computePassDesc{};
        WGPUComputePassEncoder computePass = wgpuCommandEncoderBeginComputePass(encoder, &computePassDesc);
        
        // Set compute pipeline and bind group
        wgpuComputePassEncoderSetPipeline(computePass, computePipeline);
        wgpuComputePassEncoderSetBindGroup(computePass, 0, computeBindGroup, 0, nullptr);
        
        // Dispatch compute shader
        uint32_t voxelCount = gpuVoxelManager->getVoxelCount();
        uint32_t workgroups = (voxelCount + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        
        wgpuComputePassEncoderDispatchWorkgroups(computePass, workgroups, 1, 1);
        
        wgpuComputePassEncoderEnd(computePass);
        wgpuComputePassEncoderRelease(computePass);
    }
    
    // Then, run render pass
    {
        // Begin a render pass
        WGPUColor clearColor = {0.1f, 0.2f, 0.3f, 1.0f};
        WGPURenderPassColorAttachment colorAttachment = {};
        colorAttachment.view = view;
        colorAttachment.loadOp = WGPULoadOp_Clear;
        colorAttachment.storeOp = WGPUStoreOp_Store;
        colorAttachment.clearValue = clearColor;
        
        WGPURenderPassDepthStencilAttachment depthAttachment{};
        depthAttachment.view = depthTextureView;
        depthAttachment.depthLoadOp = WGPULoadOp_Clear;
        depthAttachment.depthStoreOp = WGPUStoreOp_Store;
        depthAttachment.depthClearValue = 1.0f;
        
        WGPURenderPassDescriptor renderPassDesc{};
        renderPassDesc.colorAttachmentCount = 1;
        renderPassDesc.colorAttachments = &colorAttachment;
        renderPassDesc.depthStencilAttachment = &depthAttachment;
        
        WGPURenderPassEncoder renderPass = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDesc);
        if (!renderPass) {
            wgpuCommandEncoderRelease(encoder);
            wgpuTextureViewRelease(view);
            LogMessage("Failed to begin render pass");
            return;
        }
        
        // Set render pipeline and bind group
        wgpuRenderPassEncoderSetPipeline(renderPass, renderPipeline);
        wgpuRenderPassEncoderSetBindGroup(renderPass, 0, renderBindGroup, 0, nullptr);
        
        // Draw voxels using instanced rendering
        // Each instance is a voxel, and we use a unit cube mesh
        // 36 vertices for a cube (6 faces, 2 triangles per face, 3 vertices per triangle)
        uint32_t voxelCount = gpuVoxelManager->getVoxelCount();
        wgpuRenderPassEncoderDraw(renderPass, 36, voxelCount, 0, 0);
        
        wgpuRenderPassEncoderEnd(renderPass);
        wgpuRenderPassEncoderRelease(renderPass);
    }
    
    // Finish and submit
    WGPUCommandBufferDescriptor cmdBufferDesc = {};
    WGPUCommandBuffer cmdBuffer = wgpuCommandEncoderFinish(encoder, &cmdBufferDesc);
    if (!cmdBuffer) {
        wgpuCommandEncoderRelease(encoder);
        wgpuTextureViewRelease(view);
        LogMessage("Failed to finish command buffer");
        return;
    }
    
    wgpuQueueSubmit(queue, 1, &cmdBuffer);
    
    // Cleanup
    wgpuCommandBufferRelease(cmdBuffer);
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

// Initialize the voxel engine
bool InitializeVoxelEngine() {
    // Set up voxel system
    voxelSystem = std::make_unique<VoxelSystem>(MAX_VOXELS, 0.0f, 0.0f, 0.0f, WORLD_SIZE);
    
    // Create render pipelines
    if (!CreateRenderPipelines()) {
        LogMessage("Failed to create render pipelines");
        return false;
    }
    
    // Create example voxel data
    CreateExampleVoxels();
    
    // Initial upload to GPU
    voxelSystem->uploadToGPU(*gpuVoxelManager);
    
    // Now create bind groups after voxels are uploaded
    CreateBindGroups();
    
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
    
    // Create the swap chain
    WGPUSwapChainDescriptor swapChainDesc = {};
    swapChainDesc.usage = WGPUTextureUsage_RenderAttachment;
    swapChainDesc.format = WGPUTextureFormat_BGRA8Unorm;
    swapChainDesc.width = CANVAS_WIDTH;
    swapChainDesc.height = CANVAS_HEIGHT;
    swapChainDesc.presentMode = WGPUPresentMode_Fifo;
    
    WGPUSurfaceDescriptorFromCanvasHTMLSelector canvasDesc = {};
    canvasDesc.chain.sType = WGPUSType_SurfaceDescriptorFromCanvasHTMLSelector;
    canvasDesc.selector = "#canvas";
    
    WGPUSurfaceDescriptor surfaceDesc = {};
    surfaceDesc.nextInChain = (WGPUChainedStruct*)&canvasDesc;
    
    WGPUSurface surface = wgpuInstanceCreateSurface(wgpuCreateInstance(NULL), &surfaceDesc);
    if (!surface) {
        LogMessage("Failed to create surface");
        hasError = true;
        return;
    }
    
    swapChain = wgpuDeviceCreateSwapChain(device, surface, &swapChainDesc);
    if (!swapChain) {
        LogMessage("Failed to create swap chain");
        hasError = true;
        return;
    }
    
    // Initialize voxel engine
    if (InitializeVoxelEngine()) {
        initialized = true;
        LogMessage("Voxel engine initialized successfully");
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
        snprintf(buffer, sizeof(buffer), "Frame: %d, Initialized: %s, Has Error: %s, Voxels: %zu", 
               frameCount, 
               initialized ? "Yes" : "No",
               hasError ? "Yes" : "No",
               voxelSystem ? voxelSystem->getVoxels().size() : 0);
        LogMessage(buffer);
    }
}

// Clean up WebGPU resources
void CleanupWebGPU() {
    LogMessage("Cleaning up WebGPU resources...");
    
    // Clean up voxel engine resources
    gpuVoxelManager.reset();
    voxelSystem.reset();
    
    // Clean up render resources
    if (renderBindGroup) {
        wgpuBindGroupRelease(renderBindGroup);
        renderBindGroup = nullptr;
    }
    
    if (computeBindGroup) {
        wgpuBindGroupRelease(computeBindGroup);
        computeBindGroup = nullptr;
    }
    
    if (bindGroupLayout) {
        wgpuBindGroupLayoutRelease(bindGroupLayout);
        bindGroupLayout = nullptr;
    }
    
    if (renderPipeline) {
        wgpuRenderPipelineRelease(renderPipeline);
        renderPipeline = nullptr;
    }
    
    if (computePipeline) {
        wgpuComputePipelineRelease(computePipeline);
        computePipeline = nullptr;
    }
    
    if (cameraBuffer) {
        wgpuBufferRelease(cameraBuffer);
        cameraBuffer = nullptr;
    }
    
    if (depthTextureView) {
        wgpuTextureViewRelease(depthTextureView);
        depthTextureView = nullptr;
    }
    
    if (depthTexture) {
        wgpuTextureRelease(depthTexture);
        depthTexture = nullptr;
    }
    
    if (swapChain) {
        wgpuSwapChainRelease(swapChain);
        swapChain = nullptr;
    }
    
    if (device) {
        wgpuDeviceRelease(device);
        device = nullptr;
    }
    
    if (queue) {
        wgpuQueueRelease(queue);
        queue = nullptr;
    }
}

// Main entry point
int main() {
    LogMessage("Starting WebGPU Voxel Engine");
    
    // Initialize WebGPU
    InitWebGPU();
    
    // Start the main loop
    emscripten_set_main_loop(MainLoop, 0, 1);
    
    // Register cleanup
    atexit(CleanupWebGPU);
    
    return 0;
}