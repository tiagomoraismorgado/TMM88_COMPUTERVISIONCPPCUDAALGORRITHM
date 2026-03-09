//================================================================================
// main.cpp ── Host driver for Demokratie · Stahl · Philips · Zukunft perceptual eval
// March 2026 – Overclocked & User-Friendly Edition
//================================================================================
//
// Loads 512×512 grayscale images (reference + distorted)
// Calls GPU perceptual quality wrapper
// Measures precise wall-clock time
// Prints rich results, CUDA info, optional CSV export
//
// Build example (C++20 required):
//   nvcc -std=c++20 -O3 main.cpp demokratie_stahl_philips_zukunft_quality.cu -o dsp_quality \
//     `pkg-config --cflags --libs opencv4`
//
// Usage:
//   ./dsp_quality
//   ./dsp_quality ref.bmp dist.jp2
//   ./dsp_quality ref1.png dist1.jpg ref2.png dist2.png --verbose --csv results.csv
//   ./dsp_quality --help
//
//================================================================================
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <format>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>
#include <optional>

namespace fs = std::filesystem;

// ── Forward declaration from your .cu file
// Adjust name/signature to match your actual implementation
extern cudaError_t demokratie_stahl_philips_perceptual_eval(
    const cv::Mat& ref_512_gray,
    const cv::Mat& dist_512_gray,
    DemokratieStahlPhilipsResult& result,
    bool verbose = false
);

// ── Result structure (copied from .cu for host visibility)
struct DemokratieStahlPhilipsResult {
    float mad_score       = 0.0f;
    float detection_part  = 0.0f;
    float appearance_part = 0.0f;
    double gpu_ms         = 0.0;
};

//───────────────────────────────────────────────────────────────────────────────
// ANSI color helpers (optional – comment out if not wanted)
//───────────────────────────────────────────────────────────────────────────────
constexpr const char* CLR_RESET   = "\033[0m";
constexpr const char* CLR_BOLD    = "\033[1m";
constexpr const char* CLR_CYAN    = "\033[36m";
constexpr const char* CLR_GREEN   = "\033[32m";
constexpr const char* CLR_YELLOW  = "\033[33m";
constexpr const char* CLR_RED     = "\033[31m";

//───────────────────────────────────────────────────────────────────────────────
// Load & validate 512×512 grayscale image
//───────────────────────────────────────────────────────────────────────────────
[[nodiscard]] cv::Mat load_gray_or_die(const fs::path& path) {
    if (!fs::exists(path)) {
        std::cerr << std::format("{}Error: file not found → {}{}", CLR_RED, path.string(), CLR_RESET) << '\n';
        std::exit(1);
    }

    cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << std::format("{}Error: cannot decode image → {}{}", CLR_RED, path.string(), CLR_RESET) << '\n';
        std::exit(1);
    }

    if (img.size() != cv::Size(512, 512)) {
        std::cerr << std::format("{}Error: image must be exactly 512×512 → {} (got {}×{}){}", 
                                 CLR_RED, path.string(), img.cols, img.rows, CLR_RESET) << '\n';
        std::exit(1);
    }

    return img;
}

//───────────────────────────────────────────────────────────────────────────────
// Print CUDA device overview
//───────────────────────────────────────────────────────────────────────────────
void print_cuda_overview() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << CLR_YELLOW << "No CUDA-capable device detected." << CLR_RESET << '\n';
        return;
    }

    std::cout << CLR_CYAN << "CUDA Device Info" << CLR_RESET << '\n';
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::cout << std::format("  Using device .. : {} (id 0)\n", prop.name);
    std::cout << std::format("  Compute cap ... : {}.{}\n", prop.major, prop.minor);
    std::cout << std::format("  Global memory . : {:.1f} GiB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    std::cout << std::format("  Max threads/blk : {}\n", prop.maxThreadsPerBlock);
    std::cout << '\n';
}

//───────────────────────────────────────────────────────────────────────────────
// Evaluate one image pair and collect result + timing
//───────────────────────────────────────────────────────────────────────────────
struct EvalResult {
    std::string ref_name;
    std::string dist_name;
    DemokratieStahlPhilipsResult metrics;
    double wall_ms = 0.0;
    cudaError_t cuda_status = cudaSuccess;
};

EvalResult evaluate_pair(const cv::Mat& ref, const cv::Mat& dist,
                         const std::string& ref_name, const std::string& dist_name,
                         bool verbose = false) {
    EvalResult res{ref_name, dist_name};

    cudaDeviceSynchronize();  // ensure clean starting point
    auto t0 = std::chrono::steady_clock::now();

    res.cuda_status = demokratie_stahl_philips_perceptual_eval(ref, dist, res.metrics, verbose);

    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    res.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return res;
}

//───────────────────────────────────────────────────────────────────────────────
// Print single result nicely
//───────────────────────────────────────────────────────────────────────────────
void print_result(const EvalResult& r, bool verbose = false) {
    std::cout << CLR_BOLD << "┌────────────────────────────────────────────────────────────┐" << CLR_RESET << '\n';
    std::cout << std::format("│ {} vs {} │\n", r.ref_name, r.dist_name);
    std::cout << CLR_BOLD << "└────────────────────────────────────────────────────────────┘" << CLR_RESET << '\n';

    if (r.cuda_status != cudaSuccess) {
        std::cerr << CLR_RED << std::format("CUDA error: {}", cudaGetErrorString(r.cuda_status)) << CLR_RESET << '\n';
        return;
    }

    std::cout << std::format("  MAD-Score ............ : {:.4f}\n", r.metrics.mad_score);
    std::cout << std::format("  Detection distortion . : {:.4f}\n", r.metrics.detection_part);
    std::cout << std::format("  Appearance distortion  : {:.4f}\n", r.metrics.appearance_part);
    std::cout << std::format("  GPU kernel time ...... : {:.2f} ms\n", r.metrics.gpu_ms);
    std::cout << std::format("  Wall-clock time ...... : {:.2f} ms\n", r.wall_ms);

    if (verbose) {
        std::cout << CLR_GREEN << "  → Solid perceptual evaluation complete (Demokratie + Stahl + Philips spirit)" << CLR_RESET << '\n';
    }
    std::cout << '\n';
}

//───────────────────────────────────────────────────────────────────────────────
// Optional CSV export
//───────────────────────────────────────────────────────────────────────────────
void export_to_csv(const std::vector<EvalResult>& results, const std::string& filename) {
    std::ofstream csv(filename);
    if (!csv.is_open()) {
        std::cerr << CLR_RED << "Failed to open CSV file: " << filename << CLR_RESET << '\n';
        return;
    }

    csv << "ref_file,dist_file,mad_score,detection_part,appearance_part,gpu_ms,wall_ms,cuda_status\n";
    for (const auto& r : results) {
        csv << std::format("{},{},{:.4f},{:.4f},{:.4f},{:.2f},{:.2f},{}\n",
                           r.ref_name, r.dist_name,
                           r.metrics.mad_score, r.metrics.detection_part, r.metrics.appearance_part,
                           r.metrics.gpu_ms, r.wall_ms,
                           (r.cuda_status == cudaSuccess ? "OK" : cudaGetErrorString(r.cuda_status)));
    }
    std::cout << CLR_GREEN << std::format("Results exported to: {}", filename) << CLR_RESET << '\n';
}

//───────────────────────────────────────────────────────────────────────────────
// Print help
//───────────────────────────────────────────────────────────────────────────────
void print_help(const char* progname) {
    std::cout << CLR_CYAN << "Demokratie · Stahl · Philips · Zukunft Perceptual Quality Tool\n" << CLR_RESET;
    std::cout << "Usage:\n";
    std::cout << "  " << progname << " [ref1 dist1 [ref2 dist2 ...]] [--verbose] [--csv file.csv] [--help]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --verbose         Show detailed kernel output (if supported)\n";
    std::cout << "  --csv <file>      Export results to CSV file\n";
    std::cout << "  --help            Show this help\n\n";
    std::cout << "Default (no args): uses horse.bmp vs horse.JP2.bmp\n";
}

//───────────────────────────────────────────────────────────────────────────────
// Main
//───────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::span<char*> args(argv, static_cast<size_t>(argc));
    std::string progname = fs::path(args[0]).filename().string();

    bool verbose = false;
    std::optional<std::string> csv_path;

    std::vector<std::pair<fs::path, fs::path>> pairs;

    for (size_t i = 1; i < args.size(); ++i) {
        std::string arg = args[i];
        if (arg == "--help" || arg == "-h") {
            print_help(progname.c_str());
            return 0;
        }
        if (arg == "--verbose") {
            verbose = true;
            continue;
        }
        if (arg == "--csv" && i + 1 < args.size()) {
            csv_path = args[++i];
            continue;
        }
        // Assume image pair
        if (i + 1 < args.size() && !args[i + 1].starts_with("--")) {
            pairs.emplace_back(fs::path(arg), fs::path(args[++i]));
        }
    }

    // Default test pair if nothing provided
    if (pairs.empty()) {
        pairs.emplace_back("horse.bmp", "horse.JP2.bmp");
    }

    print_cuda_overview();

    std::cout << std::format("Evaluating {} image pair(s) {}\n\n",
                             pairs.size(), verbose ? "(verbose mode)" : "");

    std::vector<EvalResult> all_results;
    all_results.reserve(pairs.size());

    int idx = 1;
    for (const auto& [ref_p, dist_p] : pairs) {
        std::cout << std::format("{}/{} Processing pair...\n", idx++, pairs.size());
        cv::Mat ref  = load_gray_or_die(ref_p);
        cv::Mat dist = load_gray_or_die(dist_p);

        auto res = evaluate_pair(ref, dist, ref_p.filename().string(), dist_p.filename().string(), verbose);
        print_result(res, verbose);
        all_results.push_back(std::move(res));
    }

    if (csv_path) {
        export_to_csv(all_results, *csv_path);
    }

    std::cout << CLR_GREEN << CLR_BOLD << "Evaluation complete – Gemeinsam schaffen wir das! 🚀" << CLR_RESET << '\n';
    return 0;
}