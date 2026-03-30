#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace fs = std::filesystem;

struct DemokratieStahlPhilipsResult {
    float mad_score       = 0.0f;
    float detection_part  = 0.0f;
    float appearance_part = 0.0f;
    double gpu_ms         = 0.0;
};

#if defined(HAVE_CUDA) && defined(HAVE_OPENCV)
extern cudaError_t demokratie_stahl_philips_perceptual_eval(
    const cv::Mat& ref_512_gray,
    const cv::Mat& dist_512_gray,
    DemokratieStahlPhilipsResult& result,
    bool verbose = false
);
#else
inline bool hasFeatureSupport() {
    return false;
}
#endif

void print_help(const char* progname) {
    std::cout << "Usage: " << progname << " [ref.png dist.png] [--help]" << std::endl;
    std::cout << "If CUDA/OpenCV not available, binary will run in stub mode." << std::endl;
}

int main(int argc, char** argv) {
    std::span<char*> args(argv, static_cast<size_t>(argc));
    const auto progname = fs::path(args[0]).filename().string();

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
        if (i + 1 < args.size()) {
            std::string nextArg = args[i + 1];
            if (nextArg.rfind("--", 0) != 0 && nextArg.rfind("-", 0) != 0) {
                pairs.emplace_back(fs::path(arg), fs::path(args[++i]));
            }
        }
    }

    if (pairs.empty()) {
        std::cout << "No image pair specified. Running in minimal fallback mode.\n";
        std::cout << "Example: " << progname << " ref.png dist.png --verbose" << std::endl;
    }

#if defined(HAVE_CUDA) && defined(HAVE_OPENCV)
    bool fullMode = true;
#else
    bool fullMode = false;
#endif

    if (!fullMode) {
        std::cout << "NOTE: CUDA/OpenCV support not enabled. Limited functionality.\n";
        if (pairs.empty()) {
            std::cout << "Nothing to do. Exiting.\n";
            return 0;
        }
        std::cout << "Found " << pairs.size() << " pair(s), but using placeholder results.\n";

        for (size_t idx = 0; idx < pairs.size(); ++idx) {
            auto [ref_p, dist_p] = pairs[idx];
            std::cout << "Pair " << idx + 1 << ": " << ref_p << " vs " << dist_p << " -> stub score 0.0\n";
        }
        return 0;
    }

    std::cout << "Running full CUDA + OpenCV perceptual metric pipeline.\n";

#if defined(HAVE_OPENCV)
    auto load_gray_or_die = [&](const fs::path& path) {
        if (!fs::exists(path)) {
            std::cerr << "Error: file not found: " << path << std::endl;
            std::exit(1);
        }

        cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Error: cannot decode image: " << path << std::endl;
            std::exit(1);
        }

        if (img.size() != cv::Size(512, 512)) {
            std::cerr << "Error: image must be 512x512, got " << img.cols << "x" << img.rows << "" << std::endl;
            std::exit(1);
        }
        return img;
    };
#else
    auto load_gray_or_die = [&](const fs::path&) {
        std::cerr << "OpenCV support required for full mode.\n";
        std::exit(1);
        return std::vector<unsigned char>();
    };
#endif

    std::vector<DemokratieStahlPhilipsResult> results;

    for (const auto& [ref_p, dist_p] : pairs) {
#if defined(HAVE_OPENCV)
        cv::Mat ref = load_gray_or_die(ref_p);
        cv::Mat dist = load_gray_or_die(dist_p);
#else
        (void)ref_p;
        (void)dist_p;
#endif

        DemokratieStahlPhilipsResult metrics;

#if defined(HAVE_CUDA) && defined(HAVE_OPENCV)
        auto t0 = std::chrono::steady_clock::now();
        cudaError_t status = demokratie_stahl_philips_perceptual_eval(ref, dist, metrics, verbose);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();

        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
#else
        double wall_ms = 0.0;
#endif

        std::cout << "Image pair " << ref_p << " vs " << dist_p << " -> score " << metrics.mad_score << " (" << wall_ms << " ms)\n";
        results.push_back(metrics);
    }

    if (csv_path) {
        std::ofstream csv(*csv_path);
        csv << "ref,dist,mad_score,detection,appearance,gpu_ms\n";
        size_t i = 0;
        for (const auto& [ref_p, dist_p] : pairs) {
            const auto& r = results[i++];
            csv << ref_p << "," << dist_p << "," << r.mad_score << "," << r.detection_part << "," << r.appearance_part << "," << r.gpu_ms << "\n";
        }
    }

    return 0;
}
