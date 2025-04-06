#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <locale>
#include <codecvt>
#include "../neural_network/neural_network.h"

namespace file_handling {

    namespace fs = std::filesystem;

    #define PRINT(x)std::cout<<x<<std::endl;

    class FileReader {
    private:
        fs::path fpath;
        
        // Helper function to split a string by whitespace
        std::vector<std::string> split(const std::string& s) {
            std::vector<std::string> tokens;
            std::string token;
            std::istringstream tokenStream(s);
            while (tokenStream >> token) {
                tokens.push_back(token);
            }
            return tokens;
        }

        // Convert UTF-8 string to wstring (for Windows paths)
        std::wstring utf8_to_wstring(const std::string& str) {
            std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
            return converter.from_bytes(str);
        }

    public:
        // Constructor that handles both ANSI and Unicode paths
        FileReader(const std::string& path) {
            #ifdef _WIN32
            // On Windows, use wide char API for Unicode paths
            fpath = fs::path(utf8_to_wstring(path));
            #else
            // On other systems, UTF-8 paths work directly
            fpath = fs::path(path);
            #endif
            
            if (!fs::exists(fpath)) {
                throw std::runtime_error("File does not exist: " + path);
            }
        }

        /// @brief Reads the training data file and parses it into TrainingSample objects
        /// @return Vector of TrainingSample objects
        std::vector<neural_network::TrainingSample> readTrainingData() {
            std::ifstream file(fpath);
            if (!file.is_open()) {
                // throw std::runtime_error("Could not open file: " + fpath.string());
                PRINT("Could not open file: " + fpath.string())
            }

            std::vector<neural_network::TrainingSample> samples;
            std::string line;
            
            // Skip the header line (x1 x2 x3 x4 y)
            std::getline(file, line);

            while (std::getline(file, line)) {
                if (line.empty()) continue;

                auto tokens = split(line);
                if (tokens.size() < 2) continue;  // Skip malformed lines

                neural_network::TrainingSample sample;
                
                // All tokens except the last one are input features
                for (size_t i = 0; i < tokens.size() - 1; ++i) {
                    try {
                        sample.input_data.push_back(std::stod(tokens[i]));
                    } catch (const std::exception& e) {
                        // throw std::runtime_error("Error parsing number in line: " + line);
                        PRINT("Error parsing number in line: " + line)
                    }
                }

                // Last token is the expected output
                try {
                    sample.expected_output.push_back(std::stod(tokens.back()));
                } catch (const std::exception& e) {
                    // throw std::runtime_error("Error parsing expected output in line: " + line);
                    PRINT("Error parsing expected output in line: " + line)
                }

                samples.push_back(sample);
            }

            return samples;
        }

        /// @brief Checks if the file exists and is readable
        /// @return True if file is valid, false otherwise
        bool isValid() const {
            return fs::exists(fpath) && fs::is_regular_file(fpath);
        }
    };

} // namespace file_handling