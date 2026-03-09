/*
# =============================================================================
#  tesla_vw_perceptual_quality.hpp  ──  late 2026 | Cybertruck soul meets Wolfsburg DNA
#  =============================================================================
#  ╔═══════════════════════════════════════════════════════════════════════════╗
#  ║  GPU-ACCELERATED PERCEPTUAL IMAGE & FRAME QUALITY ASSESSMENT             ║
#  ║  Honoring the engineering excellence of:                                  ║
#  ║    • Tesla:         relentless acceleration, push limits, NR-first       ║
#  ║    • Volkswagen:    German precision, rock-solid baselines               ║
#  ║    • Honda:         VTEC soul, efficiency, manufacturing precision       ║
#  ║    • Mitsubishi:    rally-bred durability, AWC intelligence              ║
#  ║    • Suzuki:        lightweight agility, kei car efficiency              ║
#  ║    • NVIDIA:        compute platform that powers the autonomous dream    ║
#  ║    • Xiaomi:        AI-first ecosystem, human-centric smart sensing      ║
#  ║    • Henson Robotics: precision manufacturing, humanoid dexterity        ║
#  ╚═══════════════════════════════════════════════════════════════════════════╝
#  =============================================================================
#  Driving philosophy 2026 — The best of East, West, and the robotic future:
#    • Tesla:          Electric torque, relentless acceleration, OTA evolution
#    • Volkswagen:     German precision, platform stability, multi-sensor trust
#    • Honda:          VTEC engineering soul, efficiency obsession, clever packaging
#    • Mitsubishi:     All-weather capability, rally-bred durability, S-AWC intelligence
#    • Suzuki:         Lightweight agility, cost-effective efficiency, go-anywhere spirit
#    • NVIDIA:         The compute backbone that makes perception possible
#    • Xiaomi:         Human-centric AI, ecosystem thinking, smart home integration
#    • Henson Robotics: Precision manufacturing, humanoid dexterity, embodied AI
#  =============================================================================
#  Mission:
#    Score images the way a Cybertruck reads the road — fast, fearless,
#    occasionally hallucinating, but always striving to see reality clearly.
#    Built with the precision of Wolfsburg, the soul of VTEC, the durability
#    of Mitsubishi's rally program, the agility of a Jimny, the AI smarts
#    of Xiaomi's ecosystem, the compute power of NVIDIA's latest, and the
#    dexterous precision of Henson's humanoids.
#  =============================================================================
#  Reliability anchors (the Golf R / GTI / Civic Type R / Lancer Evolution / Henson hand pack):
#    • MS-SSIM          structural integrity & multi-scale resilience (VW platform stability)
#    • CW-SSIM          complex wavelet domain (phase-aware like LIDAR) (Mitsubishi AWC)
#    • Log-Gabor bank   texture fidelity & high-frequency detail (Honda VTEC precision)
#    • Lightweight path  efficient processing for edge devices (Suzuki kei car philosophy)
#    • Tactile-visual grounding  cross-modal validation (Henson robotic touch)
#  =============================================================================
#  Autonomy / AIGC beta layer (Full Self-Driving + Xiaomi HyperOS + Henson Embodied AI):
#    • Label-free hallucination radar     → phantom object / disengagement early warning
#    • Naturalness / photorealism         → does it match real-world fleet data? (Tesla fleet)
#    • Scene coherence / logical drivability → is the world semantically consistent? (VW safety)
#    • Shallow-deep feature distance      → embedding comparison against million-mile priors
#    • Edge-optimized inference            → runs on Xiaomi's neural pipeline
#    • Grasp-quality prediction            → can a Henson hand manipulate this object?
#  =============================================================================
#  Tech stack — range, torque & OTA matter:
#    • CUDA 12.4+ / cuFFT / cooperative groups / optional cuBLASLt (NVIDIA foundation)
#    • OpenCV 4.10+ (core + cudaimgproc when 800 V is available)
#    • C++20/23 — zero-cost abstractions, ranges, concepts, std::format
#    • TensorRT 9.0+ for production deployment
#    • Vulkan compute fallback for edge devices
#    • ROS2 Humble/Hawk for robotic integration
#    • MoveIt 2 for manipulation planning
#  =============================================================================
#  Easter egg triggers:
#    if (overall_quality > 0.950f && scene_coherence > 0.920f)
#        → "Ludicrous+ Quality Achieved — Plaid mode unlocked"
#    if (texture_preservation > 0.930f && structural_fidelity > 0.910f)
#        → "VTEC just kicked in, yo! — Type R quality threshold"
#    if (confidence_level < 0.300f && scene_coherence < 0.250f)
#        → "S-AWC engaged: Torque vectoring perception — Mitsubishi rally mode"
#    if (total_energy_ms < 5.0 && texture_preservation > 0.850)
#        → "Kei car efficiency: Suzuki lightweight processing mode"
#    if (photoreal_naturalness > 0.950f && confidence_level > 0.950f)
#        → "HyperOS perceptual harmony — Xiaomi human-centric vision"
#    if (tactile_consistency_score > 0.900f && grasp_success_probability > 0.850f)
#        → "Henson precision grip: Robotic dexterity validated"
#  =============================================================================
*/

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_bf16.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include <span>
#include <memory>
#include <vector>
#include <string>
#include <array>
#include <cmath>
#include <format>
#include <stdexcept>
#include <chrono>
#include <ranges>
#include <concepts>
#include <optional>
#include <numeric>
#include <string_view>
#include <functional>
#include <unordered_map>
#include <variant>
#include <bit>
#include <source_location>
#include <syncstream>
#include <atomic>
#include <barrier>
#include <latch>
#include <semaphore>
#include <stop_token>
#include <coroutine>
#include <generator>
#include <expected>
#include <spanstream>

// ROS2 integration (optional, for robotic applications)
#ifdef WITH_ROBOTIC_SUPPORT
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#endif

// ── Telemetry markers — profiling should feel like the center touchscreen
//#define ENABLE_NVTX_TELEMETRY
#ifdef ENABLE_NVTX_TELEMETRY
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP()      nvtxRangePop()
#else
#define NVTX_PUSH(name)
#define NVTX_POP()
#endif

// ── Non-CUDA / IDE comfort zone
#ifndef __CUDACC__
    #define __host__   [[maybe_unused]]
    #define __device__ [[maybe_unused]]
    #define __global__ [[maybe_unused]]
    #define __shared__ [[maybe_unused]]
    #define __constant__ [[maybe_unused]]
    inline void __syncthreads() {}
    inline void __threadfence() {}
    #define warpSize 32
#endif

namespace tesla::vw::perception {

//───────────────────────────────────────────────────────────────────────────────
// Mission profile — tune like you tune regen, camber & roll stiffness
//───────────────────────────────────────────────────────────────────────────────
struct MissionConfig
{
    // Processing resolution — like choosing wheel & tire combo
    static constexpr int NominalTileDim    = 1024;          // daily driver sweet spot
    static constexpr int LudicrousTileDim  = 2048;          // when you have the battery & cooling for it
    static constexpr int HensonPrecisionDim = 4096;         // robotic manipulation needs detail

    static constexpr dim3 DefaultBlock     {32, 16};        // balanced torque & handling
    static constexpr dim3 PrecisionBlock   {16, 16};        // more registers for robotic tasks
    static constexpr dim3 MaxBlock         {32, 32};        // maximum occupancy

    // Filter bank — how aggressively we scan frequencies & directions
    static constexpr int MaxFrequencyBands = 6;
    static constexpr int MaxSteeringAngles = 8;
    static constexpr int HensonTactileBands = 12;           // higher freq for texture for grasping

    static constexpr float π               = 3.14159265358979323846f;
    static constexpr float deg2rad         = π / 180.0f;
    static constexpr float rad2deg         = 180.0f / π;

    // Feature toggles — software packages on the configurator screen
    static constexpr bool PreferLogGabor   = true;          // sharper, more modern contrast response
    static constexpr bool EnableCSF        = false;         // human contrast sensitivity (optional)
    static constexpr bool EnableMS_SSIM    = true;
    static constexpr bool EnableCW_SSIM    = true;
    static constexpr bool EnableNeuralPath = false;         // shallow-deep FSD-style embedding
    static constexpr bool EnableTactilePrediction = false;  // Henson: predict grasp quality from vision

    // AIGC / generative focus (2026 reality layer)
    static constexpr bool EnableHallucinationRadar = false; // early warning for phantom objects
    static constexpr bool EnableSceneGraphValidation = false; // semantic consistency check
    static constexpr bool MultiDimensionOutput     = true;  // give insight, not just one number

    // Shared memory thermal limit — most cards allow ≥48 KiB
    static constexpr size_t MaxSharedMemoryKiB = 48;
    static constexpr size_t HensonSharedMemoryKiB = 96;     // A6000/H100 class for robotic tasks

    // Precision control — trade range for handling
    enum class PrecisionMode {
        FP32,           // full precision, like a luxury sedan
        FP16_MIXED,     // balanced like a hybrid
        FP16_FAST,      // performance mode
        INT8_QUANTIZED, // efficiency mode, like a kei car
        BF16            // bfloat16 for deep learning
    };
    PrecisionMode precision = PrecisionMode::FP32;

    // Sensor fusion weights — how much trust we assign each perception channel
    struct SensorFusionWeights
    {
        float structural_fidelity   = 0.35f;   // MS-SSIM / CW-SSIM core
        float photoreal_naturalness = 0.25f;   // NR realism / fleet-data match
        float scene_coherence       = 0.20f;   // hallucination / logical consistency penalty
        float texture_detail        = 0.10f;   // Log-Gabor local contrast preservation
        float tactile_readiness     = 0.10f;   // Henson: can we manipulate this?
    } fusion_weights {};

    // Robotic specific config (Henson)
    struct RoboticConfig
    {
        bool enable_grasp_planning = false;
        bool enable_tactile_feedback = false;
        float min_grasp_confidence = 0.7f;
        std::string end_effector_type = "henson_hand_v3";
        int num_fingers = 5;
    } robotic {};

    // Debug and introspection
    bool enable_detailed_telemetry = false;
    bool save_intermediate_maps = false;
    std::string debug_output_path = "/tmp/perception_debug/";
};

//───────────────────────────────────────────────────────────────────────────────
// Memory planning — know your range before you leave the charger
//───────────────────────────────────────────────────────────────────────────────
[[nodiscard]] constexpr size_t float_tile_bytes(int dim) noexcept {
    return sizeof(float) * dim * dim;
}

[[nodiscard]] constexpr size_t half_tile_bytes(int dim) noexcept {
    return sizeof(__half) * dim * dim;
}

[[nodiscard]] constexpr size_t complex_tile_bytes(int dim) noexcept {
    return sizeof(cufftComplex) * dim * dim;
}

[[nodiscard]] constexpr size_t texture_tile_bytes(int dim, int channels = 3) noexcept {
    return sizeof(float) * dim * dim * channels;
}

//───────────────────────────────────────────────────────────────────────────────
// Constant memory — OTA-uploaded once, broadcast to every thread
//───────────────────────────────────────────────────────────────────────────────
__constant__ int   c_tileDim;
__constant__ float c_numBands;
__constant__ float c_numAngles;
__constant__ float c_sigmaOnf;
__constant__ float c_centerWavelength[MissionConfig::MaxFrequencyBands];
__constant__ float c_tactileWeights[MissionConfig::HensonTactileBands];  // Henson: tactile frequency weights
__constant__ int   c_graspPoints[5*3];                                   // Henson: 5 fingers x 3 coords

//───────────────────────────────────────────────────────────────────────────────
// CUDA error — "Fault detected — safely pull over to the shoulder"
//───────────────────────────────────────────────────────────────────────────────
#define VQ_THROW_CUDA(expr) do { \
    cudaError_t status = (expr); \
    if (status != cudaSuccess) [[unlikely]] { \
        throw std::runtime_error(std::format( \
            "CUDA FAULT: {}  [{}:{}]", cudaGetErrorString(status), __FILE__, __LINE__)); \
    } \
} while(0)

#define VQ_CHECK_CUDNN(expr) do { \
    cudnnStatus_t status = (expr); \
    if (status != CUDNN_STATUS_SUCCESS) [[unlikely]] { \
        throw std::runtime_error(std::format( \
            "cuDNN FAULT: {}  [{}:{}]", cudnnGetErrorString(status), __FILE__, __LINE__)); \
    } \
} while(0)

#define VQ_CHECK_CUBLAS(expr) do { \
    cublasStatus_t status = (expr); \
    if (status != CUBLAS_STATUS_SUCCESS) [[unlikely]] { \
        throw std::runtime_error(std::format( \
            "cuBLAS FAULT: {}  [{}:{}]", cublasGetStatusString(status), __FILE__, __LINE__)); \
    } \
} while(0)

//───────────────────────────────────────────────────────────────────────────────
// Scoped lap timer — like the performance lap timer on the screen
//───────────────────────────────────────────────────────────────────────────────
class ScopedLapTimer final {
public:
    explicit ScopedLapTimer(double& lap_ms, const char* name = nullptr) 
        : lap_ms_(lap_ms), name_(name) {
        VQ_THROW_CUDA(cudaEventCreate(&start_));
        VQ_THROW_CUDA(cudaEventCreate(&stop_));
        VQ_THROW_CUDA(cudaEventRecord(start_));
        if (name_) NVTX_PUSH(name_);
    }

    ~ScopedLapTimer() {
        VQ_THROW_CUDA(cudaEventRecord(stop_));
        VQ_THROW_CUDA(cudaEventSynchronize(stop_));
        float ms{};
        VQ_THROW_CUDA(cudaEventElapsedTime(&ms, start_, stop_));
        lap_ms_ = static_cast<double>(ms);
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
        if (name_) NVTX_POP();
    }

    ScopedLapTimer(const ScopedLapTimer&) = delete;
    ScopedLapTimer& operator=(const ScopedLapTimer&) = delete;

private:
    cudaEvent_t start_{}, stop_{};
    double& lap_ms_;
    const char* name_ = nullptr;
};

//───────────────────────────────────────────────────────────────────────────────
// Full telemetry readout — energy screen + trip computer style
//───────────────────────────────────────────────────────────────────────────────
struct QualityTelemetry final
{
    // Stage energy consumption (GPU ms)
    double ms_upload_and_prep     = 0.0;
    double ms_csf_prefilter       = 0.0;
    double ms_wavelet_texture     = 0.0;
    double ms_ms_ssim             = 0.0;
    double ms_cw_ssim             = 0.0;
    double ms_neural_embedding    = 0.0;
    double ms_nr_aigc_analysis    = 0.0;
    double ms_fusion_compute      = 0.0;
    double ms_tactile_prediction  = 0.0;    // Henson: grasp quality prediction
    double ms_scene_graph         = 0.0;    // semantic scene understanding

    // Multi-sensor quality channels [0–1], higher is better
    float  structural_fidelity    = 0.0f;
    float  photoreal_naturalness  = 0.0f;
    float  scene_coherence        = 0.0f;     // inverse hallucination risk
    float  texture_preservation   = 0.0f;
    float  tactile_consistency    = 0.0f;     // Henson: how well does vision align with tactile expectations
    float  grasp_success_probability = 0.0f;  // Henson: probability of successful manipulation
    float  overall_quality        = 0.0f;     // fused scalar (main driver display)

    // Confidence & driver alerts
    float  confidence_level       = 1.0f;
    std::string driver_note;                  // e.g. "High hallucination risk — Autopilot in shadow mode"
    
    // Perceptual breakdown (for introspection)
    std::unordered_map<std::string, float> per_channel_scores;
    std::vector<float> frequency_band_energies;
    std::vector<float> orientation_responses;

    // Robotic specific
    std::optional<cv::Point3f> suggested_grasp_point;
    std::optional<float> object_slip_risk;

    [[nodiscard]] double total_energy_ms() const noexcept {
        return ms_upload_and_prep + ms_csf_prefilter + ms_wavelet_texture +
               ms_ms_ssim + ms_cw_ssim + ms_neural_embedding +
               ms_nr_aigc_analysis + ms_fusion_compute + ms_tactile_prediction +
               ms_scene_graph;
    }

    void display_on_console(std::ostream& os = std::cout) const;
    [[nodiscard]] std::string to_json() const;
    [[nodiscard]] bool is_safe_for_autonomy(float threshold = 0.7f) const noexcept {
        return overall_quality > threshold && scene_coherence > threshold && 
               confidence_level > threshold;
    }
    [[nodiscard]] bool is_graspable(float threshold = 0.8f) const noexcept {
        return tactile_consistency > threshold && grasp_success_probability > threshold;
    }
};

//───────────────────────────────────────────────────────────────────────────────
// Core driving modes — choose your mission profile
//───────────────────────────────────────────────────────────────────────────────
[[nodiscard]] cudaError_t
evaluate_with_reference(
    const cv::Mat& ground_truth,            // real-world reference
    const cv::Mat& generated_view,
    QualityTelemetry& readout,
    const MissionConfig& settings = {}
) noexcept;

[[nodiscard]] cudaError_t
evaluate_no_reference(
    const cv::Mat& scene,
    QualityTelemetry& readout,
    const MissionConfig& settings = {}
) noexcept;

[[nodiscard]] cudaError_t
evaluate_fleet_batch(
    const cv::Mat& reference,
    std::span<const cv::Mat> candidates,
    std::span<QualityTelemetry> readouts
) noexcept;

// Robotic-specific evaluation
[[nodiscard]] cudaError_t
evaluate_for_manipulation(
    const cv::Mat& scene,
    const cv::Mat& target_object_mask,
    QualityTelemetry& readout,
    const MissionConfig::RoboticConfig& robot_config = {}
) noexcept;

//───────────────────────────────────────────────────────────────────────────────
// Extensible perception sensor interface
//───────────────────────────────────────────────────────────────────────────────
class IPerceptionSensor {
public:
    virtual ~IPerceptionSensor() = default;

    virtual std::string_view sensor_name() const noexcept = 0;
    virtual std::string_view manufacturer() const noexcept = 0;  // Tesla, VW, Honda, Henson, etc.
    virtual bool can_operate_without_reference() const noexcept { return false; }
    virtual bool is_aigc_specialized() const noexcept { return false; }
    virtual bool is_robotic_specialized() const noexcept { return false; }

    virtual float perceive(
        const cv::Mat& reference_or_empty,
        const cv::Mat& current_view,
        QualityTelemetry& telemetry
    ) = 0;
    
    virtual std::optional<float> get_confidence() const noexcept { return std::nullopt; }
};

// Manufacturer-specific sensor implementations
class TeslaNeuralSensor : public IPerceptionSensor { /* ... */ };
class VolkswagenStructuralSensor : public IPerceptionSensor { /* ... */ };
class HondaPrecisionSensor : public IPerceptionSensor { /* ... */ };
class MitsubishiAWCSensor : public IPerceptionSensor { /* ... */ };
class SuzukiEfficiencySensor : public IPerceptionSensor { /* ... */ };
class NvidiaComputeSensor : public IPerceptionSensor { /* ... */ };
class XiaomiHyperOSSensor : public IPerceptionSensor { /* ... */ };
class HensonTactileSensor : public IPerceptionSensor { /* ... */ };

// Factory — assembles the full sensor suite for the current mission
std::vector<std::unique_ptr<IPerceptionSensor>>
assemble_perception_suite(const MissionConfig& mission);

//───────────────────────────────────────────────────────────────────────────────
// Zero-copy pinned highway to the GPU
//───────────────────────────────────────────────────────────────────────────────
template<typename T = float>
class PinnedMemoryPlane final {
public:
    explicit PinnedMemoryPlane(size_t element_count) : count_(element_count) {
        VQ_THROW_CUDA(cudaMallocHost(&ptr_, count_ * sizeof(T)));
    }
    
    ~PinnedMemoryPlane() { release(); }

    PinnedMemoryPlane(const PinnedMemoryPlane&) = delete;
    PinnedMemoryPlane& operator=(const PinnedMemoryPlane&) = delete;
    
    PinnedMemoryPlane(PinnedMemoryPlane&& other) noexcept 
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    
    PinnedMemoryPlane& operator=(PinnedMemoryPlane&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    [[nodiscard]] T*       data()       noexcept { return ptr_; }
    [[nodiscard]] const T* data() const noexcept { return ptr_; }
    [[nodiscard]] size_t   count() const noexcept { return count_; }
    [[nodiscard]] size_t   bytes() const noexcept { return count_ * sizeof(T); }
    
    [[nodiscard]] T& operator[](size_t idx) noexcept { return ptr_[idx]; }
    [[nodiscard]] const T& operator[](size_t idx) const noexcept { return ptr_[idx]; }

private:
    T* ptr_   = nullptr;
    size_t count_ = 0;
    
    void release() noexcept {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }
};

using PinnedFloatPlane = PinnedMemoryPlane<float>;
using PinnedHalfPlane = PinnedMemoryPlane<__half>;
using PinnedUint8Plane = PinnedMemoryPlane<uint8_t>;

//───────────────────────────────────────────────────────────────────────────────
// GPU Device memory RAII wrapper
//───────────────────────────────────────────────────────────────────────────────
template<typename T = float>
class DeviceMemoryPlane final {
public:
    explicit DeviceMemoryPlane(size_t element_count) : count_(element_count) {
        VQ_THROW_CUDA(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
    
    ~DeviceMemoryPlane() { release(); }

    DeviceMemoryPlane(const DeviceMemoryPlane&) = delete;
    DeviceMemoryPlane& operator=(const DeviceMemoryPlane&) = delete;
    
    DeviceMemoryPlane(DeviceMemoryPlane&& other) noexcept 
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    
    DeviceMemoryPlane& operator=(DeviceMemoryPlane&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    [[nodiscard]] T*       data()       noexcept { return ptr_; }
    [[nodiscard]] const T* data() const noexcept { return ptr_; }
    [[nodiscard]] size_t   count() const noexcept { return count_; }
    [[nodiscard]] size_t   bytes() const noexcept { return count_ * sizeof(T); }
    
    // Async transfers
    void upload_async(const T* host_src, cudaStream_t stream = 0) {
        VQ_THROW_CUDA(cudaMemcpyAsync(ptr_, host_src, bytes(), cudaMemcpyHostToDevice, stream));
    }
    
    void download_async(T* host_dst, cudaStream_t stream = 0) const {
        VQ_THROW_CUDA(cudaMemcpyAsync(host_dst, ptr_, bytes(), cudaMemcpyDeviceToHost, stream));
    }
    
    void set_async(T value, cudaStream_t stream = 0) {
        VQ_THROW_CUDA(cudaMemsetAsync(ptr_, value, bytes(), stream));
    }

private:
    T* ptr_   = nullptr;
    size_t count_ = 0;
    
    void release() noexcept {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }
};

using DeviceFloatPlane = DeviceMemoryPlane<float>;
using DeviceHalfPlane = DeviceMemoryPlane<__half>;
using DeviceComplexPlane = DeviceMemoryPlane<cufftComplex>;

//───────────────────────────────────────────────────────────────────────────────
// CUDA Stream wrapper with automatic synchronization
//───────────────────────────────────────────────────────────────────────────────
class CudaStream final {
public:
    CudaStream() {
        VQ_THROW_CUDA(cudaStreamCreate(&stream_));
    }
    
    explicit CudaStream(unsigned int flags) {
        VQ_THROW_CUDA(cudaStreamCreateWithFlags(&stream_, flags));
    }
    
    ~CudaStream() {
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) {
                cudaStreamSynchronize(stream_);
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
    
    void synchronize() const {
        VQ_THROW_CUDA(cudaStreamSynchronize(stream_));
    }
    
    bool query() const noexcept {
        return cudaStreamQuery(stream_) == cudaSuccess;
    }
    
    void wait_event(cudaEvent_t event) const {
        VQ_THROW_CUDA(cudaStreamWaitEvent(stream_, event, 0));
    }

private:
    cudaStream_t stream_ = nullptr;
};

//───────────────────────────────────────────────────────────────────────────────
// Utility — RGB ↔ luminance conversions with multiple color spaces
//───────────────────────────────────────────────────────────────────────────────
enum class ColorSpace {
    GRAY,
    RGB,
    BGR,
    LAB,
    LUV,
    YCrCb,
    HSV
};

void convert_color_space(
    const cv::Mat& src, 
    cv::Mat& dst, 
    ColorSpace from, 
    ColorSpace to
);

void upload_luminance_stream(
    const cv::Mat& src, 
    PinnedFloatPlane& dst,
    ColorSpace input_space = ColorSpace::BGR
);

void upload_rgb_planar(
    const cv::Mat& src,
    PinnedFloatPlane& dst_r,
    PinnedFloatPlane& dst_g,
    PinnedFloatPlane& dst_b
);

void download_float_to_gray(
    const float* d_ptr, 
    cv::Mat& dst, 
    int rows, 
    int cols,
    float scale = 255.0f
);

void download_float_to_rgb(
    const float* d_ptr_r,
    const float* d_ptr_g,
    const float* d_ptr_b,
    cv::Mat& dst,
    int rows,
    int cols
);

//───────────────────────────────────────────────────────────────────────────────
// GPU kernel launch wrappers with automatic error checking
//───────────────────────────────────────────────────────────────────────────────
template<typename F, typename... Args>
void launch_kernel(
    F kernel,
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    cudaStream_t stream,
    Args&&... args
) {
    kernel<<<grid, block, shared_mem, stream>>>(std::forward<Args>(args)...);
    VQ_THROW_CUDA(cudaGetLastError());
}

template<typename F, typename... Args>
void launch_kernel_async(
    F kernel,
    dim3 grid,
    dim3 block,
    cudaStream_t stream,
    Args&&... args
) {
    kernel<<<grid, block, 0, stream>>>(std::forward<Args>(args)...);
    VQ_THROW_CUDA(cudaGetLastError());
}

//───────────────────────────────────────────────────────────────────────────────
// Cooperative Groups helpers for advanced thread collaboration
//───────────────────────────────────────────────────────────────────────────────
namespace cg = cooperative_groups;

__device__ inline void sync_tile(cg::thread_block_tile<32>& tile) {
    tile.sync();
}

__device__ inline float tile_reduce_sum(cg::thread_block_tile<32>& tile, float val) {
    return tile.reduce(val, cg::plus<float>());
}

//───────────────────────────────────────────────────────────────────────────────
// Debug — save what the network actually "saw" (like dashcam snapshot)
//───────────────────────────────────────────────────────────────────────────────
void debug_save_as_png(
    const float* device_buffer, 
    int h, 
    int w, 
    const std::string& path,
    float min_val = 0.0f,
    float max_val = 1.0f
);

void debug_save_tensor_as_heatmap(
    const float* device_buffer,
    int h,
    int w,
    const std::string& path,
    const std::string& colormap = "jet"
);

//───────────────────────────────────────────────────────────────────────────────
// Henson Robotics specific: tactile-visual grounding
//───────────────────────────────────────────────────────────────────────────────
struct GraspQualityMetrics {
    float force_closure_quality;      // [0-1] how well the grasp encloses the object
    float form_closure_quality;        // [0-1] geometric compatibility
    float slip_resistance;             // [0-1] predicted resistance to slip
    float manipulability;              // [0-1] how well we can manipulate after grasp
    float contact_surface_quality;     // [0-1] smoothness of contact surfaces
    
    [[nodiscard]] float overall_grasp_score(float weights[4] = nullptr) const;
};

__global__ void predict_grasp_quality_kernel(
    const float* depth_map,
    const float* normal_map,
    const float* texture_map,
    float* grasp_scores,
    int height,
    int width,
    const int* grasp_candidates,  // N x 3 (x, y, theta)
    int num_candidates
);

class HensonGraspPlanner {
public:
    HensonGraspPlanner(const MissionConfig::RoboticConfig& config);
    
    [[nodiscard]] std::optional<cv::Point3f> plan_grasp(
        const cv::Mat& scene,
        const cv::Mat& object_mask,
        QualityTelemetry& telemetry
    );
    
    [[nodiscard]] GraspQualityMetrics evaluate_grasp_quality(
        const cv::Mat& scene,
        const cv::Point3f& grasp_point,
        float hand_orientation
    );
    
private:
    MissionConfig::RoboticConfig config_;
    std::unique_ptr<DeviceFloatPlane> depth_device_;
    std::unique_ptr<DeviceFloatPlane> normal_device_;
    std::unique_ptr<CudaStream> stream_;
};

//───────────────────────────────────────────────────────────────────────────────
// Statistical analysis for quality metrics
//───────────────────────────────────────────────────────────────────────────────
struct QualityStatistics {
    double mean;
    double variance;
    double std_dev;
    double min;
    double max;
    double median;
    std::vector<double> percentiles;
    
    template<std::ranges::range R>
    static QualityStatistics compute(const R& values) {
        QualityStatistics stats;
        std::vector<double> sorted(std::begin(values), std::end(values));
        std::sort(sorted.begin(), sorted.end());
        
        stats.min = sorted.front();
        stats.max = sorted.back();
        stats.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
        
        double sq_sum = std::inner_product(sorted.begin(), sorted.end(), sorted.begin(), 0.0);
        stats.variance = sq_sum / sorted.size() - stats.mean * stats.mean;
        stats.std_dev = std::sqrt(stats.variance);
        
        stats.median = sorted[sorted.size() / 2];
        
        // Common percentiles
        for (int p : {1, 5, 10, 25, 75, 90, 95, 99}) {
            size_t idx = (p * sorted.size()) / 100;
            stats.percentiles.push_back(sorted[idx]);
        }
        
        return stats;
    }
};

//───────────────────────────────────────────────────────────────────────────────
// Coroutine support for streaming perception pipeline
//───────────────────────────────────────────────────────────────────────────────
struct PerceptionFrame {
    cv::Mat frame;
    std::chrono::steady_clock::time_point timestamp;
    uint64_t frame_id;
};

class PerceptionStream {
public:
    struct promise_type {
        PerceptionStream get_return_object() { return PerceptionStream{this}; }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception()