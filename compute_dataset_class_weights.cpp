#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <chrono>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <iomanip>
#include <algorithm>
#include <cstring>

namespace fs = std::filesystem;

bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

std::vector<double> weights_log(const std::vector<double>& class_freq, int num_classes) {
    std::vector<double> weights;
    double sum_w = 0.0;
    for (double freq : class_freq) {
        double w = 1.0 / std::log1p(freq);
        weights.push_back(w);
        sum_w += w;
    }
    double norm_factor = static_cast<double>(num_classes);
    for (double& w : weights) {
        w = norm_factor * w / sum_w;
    }
    return weights;
}

int main(int argc, char* argv[]) {
    bool use_custom_path = false;
    int num_classes = 19;
    std::string dataset_dir = "/path/to/cityscapes/gtFine/train/";
    std::string suffix = "_gtFine_labelTrainIds.png";

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--use_custom_path") == 0) {
            use_custom_path = true;
        } else if (std::strcmp(argv[i], "--num_classes") == 0 && i + 1 < argc) {
            num_classes = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--dataset_dir") == 0 && i + 1 < argc) {
            dataset_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--suffix") == 0 && i + 1 < argc) {
            suffix = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            return 1;
        }
    }

    std::vector<double> freqs;

    if (use_custom_path) {
        // Find all relevant .png files recursively
        std::vector<std::string> label_files;
        for (const auto& entry : fs::recursive_directory_iterator(dataset_dir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (ends_with(filename, suffix)) {
                    label_files.push_back(entry.path().string());
                }
            }
        }

        size_t file_count = label_files.size();
        if (file_count == 0) {
            throw std::runtime_error("No label files found in the directory with the given suffix.");
        }

        // Initialize count for each class using an array for efficiency
        std::array<uint64_t, 256> total_counts{};
        
        // Progress bar variables
        auto start_time = std::chrono::steady_clock::now();
        double avg_time_per_image = 0.0;
        size_t processed = 0;

        // Loop through each label image
        for (const auto& file : label_files) {
            auto image_start = std::chrono::steady_clock::now(); // Time this image

            cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                // To keep progress accurate, still update processed and avg (though time=0)
                auto image_end = std::chrono::steady_clock::now();
                double image_time = std::chrono::duration<double>(image_end - image_start).count();
                avg_time_per_image = (avg_time_per_image * processed + image_time) / (processed + 1);
                ++processed;
                // Update progress below
            } else {
                for (int r = 0; r < img.rows; ++r) {
                    const uchar* ptr = img.ptr<uchar>(r);
                    for (int c = 0; c < img.cols; ++c) {
                        ++total_counts[ptr[c]];
                    }
                }
                auto image_end = std::chrono::steady_clock::now();
                double image_time = std::chrono::duration<double>(image_end - image_start).count();
                avg_time_per_image = (avg_time_per_image * processed + image_time) / (processed + 1);
                ++processed;
            }

            // Calculate progress and ETA
            double progress = static_cast<double>(processed) / file_count;
            double eta_seconds = avg_time_per_image * (file_count - processed);
            int eta_min = static_cast<int>(eta_seconds / 60);
            int eta_sec = static_cast<int>(eta_seconds) % 60;

            // Progress bar (fixed width, e.g., 20 characters)
            int bar_width = 20;
            int filled = static_cast<int>(bar_width * progress);
            std::string bar(filled, '=');
            bar += std::string(bar_width - filled, ' ');

            // Output progress
            std::cout << "\r[" << bar << "] " 
                      << std::fixed << std::setprecision(1) << progress * 100 << "% "
                      << processed << "/" << file_count
                      << ", ETA: " << eta_min << "m " << eta_sec << "s" << std::flush;
        }
        std::cout << std::endl;

        // Print class counts (excluding class_255)
        std::cout << "Class counts: {";
        bool first = true;
        for (int u = 0; u < 256; ++u) {
            if (total_counts[u] > 0 && u != 255) {
                if (!first) std::cout << ", ";
                std::cout << "class_" << u << ": " << total_counts[u];
                first = false;
            }
        }
        std::cout << "}" << std::endl;

        // Collect frequencies excluding class_255 and zero counts
        for (int u = 0; u < num_classes; ++u) {
            if (total_counts[u] > 0) {
                freqs.push_back(static_cast<double>(total_counts[u]));
            }
        }
    } else {
        std::vector<double> cityscapes_per_class_pixel_count = {
            2.01e+9, 2.98e+8, 9.96e+8, 3.39e+7, 4.50e+7, 6.54e+7,
            9.57e+7, 2.62e+7, 7.21e+8, 5.92e+7, 1.45e+8, 8.21e+7,
            1.00e+7, 4.13e+8, 1.45e+7, 1.28e+7, 1.45e+7, 5.64e+6, 2.57e+7
        };
        freqs = cityscapes_per_class_pixel_count;

        // Print class counts for hardcoded
        std::cout << "Class counts: {";
        bool first = true;
        for (size_t i = 0; i < freqs.size(); ++i) {
            if (!first) std::cout << ", ";
            std::cout << "class_" << i << ": " << freqs[i];
            first = false;
        }
        std::cout << "}" << std::endl;
    }

    // Calculate log weights
    std::vector<double> weights = weights_log(freqs, num_classes);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Class weights: [";
    bool first = true;
    for (double w : weights) {
        if (!first) std::cout << ", ";
        std::cout << w;
        first = false;
    }
    std::cout << "]" << std::endl;

    return 0;
}