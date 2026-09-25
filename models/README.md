# models

Put the generation model here and name it `llm.gguf`, or point `llm_path` in `config.json` at another file on the tablet.

The file this stack is set up for is Qwen2.5-3B-Instruct Q4_K_M (1.8 GB), saved here as `llm.gguf`. That fits an 8 GB or 16 GB Getac. A 7B Q4 file is about 4.7 GB and will be slow on a 15 W Intel CPU. A 70B model does not fit.

The app does not download this file. Copy it onto the tablet before you turn the radios off.

`gpu_layers` stays 0 unless you brought a llama.cpp build that can use the Intel GPU. The supported path is the CPU binary.
