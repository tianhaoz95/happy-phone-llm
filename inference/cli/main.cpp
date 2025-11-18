#include <iostream>
#include <string>
#include <vector>

#include "engine/engine.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <prompt> [max_tokens]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string prompt = argv[2];
    int max_tokens = 50; // Default max tokens
    if (argc > 3) {
        try {
            max_tokens = std::stoi(argv[3]);
        } catch (const std::exception& e) {
            std::cerr << "Invalid max_tokens value: " << argv[3] << ". Using default of 50." << std::endl;
        }
    }

    engine::LLM llm;
    if (!llm.load_model(model_path)) {
        std::cerr << "Failed to load model from " << model_path << std::endl;
        return 1;
    }

    std::string generated_text = llm.generate(prompt, max_tokens);
    std::cout << "Generated Text:\n" << generated_text << std::endl;

    return 0;
}
