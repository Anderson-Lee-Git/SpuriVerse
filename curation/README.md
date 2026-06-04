# Vision Language Model Interface Documentation

This document details the integration methods for each supported model type in the VLMInterface.

## Model Integrations

### Gemini Models
- Uses Google's GenerativeAI SDK
- Configured with GenerationConfig for temperature, top_p, and token limits
- Generates content using model.generate_content() with image and text inputs

### LLaVA Models
- LLaVA 1.6: Uses LlavaNextProcessor and LlavaNextForConditionalGeneration from HuggingFace
- LLaVA 1.5: Uses AutoProcessor and LlavaForConditionalGeneration from HuggingFace
- Both run inference using model.generate() with processor-prepared inputs

### LLaMA Models
- Uses FastVisionModel from unsloth library
- Supports Llama-3.2-11B and Llama-3.2-90B variants
- Processes inputs using tokenizer.apply_chat_template()
- Generates responses using model.generate() with 4-bit quantization

### GPT Models (OpenAI)
- GPT-4o: Uses chat.completions.create API with model "gpt-4o-2024-08-06"
- GPT-4o-mini: Uses chat.completions.create API with model "gpt-4o-mini-2024-07-18"
- Both accept base64 encoded images via image_url in messages

### OpenAI Vision Models
- O4-mini: Uses responses.create API with "o4-mini" model
- O3: Uses responses.create API with "o3" model
- Both support input_image and input_text in content structure
- Include reasoning effort parameter and higher token limits (3000)

### Qwen Models
- Qwen-vl-max/plus: Uses OpenAI-compatible API through Dashscope
- Qwen-2.5 series (7B/32B/72B): Uses FastVisionModel from unsloth
- Cloud variants use chat.completions.create API
- Local variants use model.generate() with tokenizer preprocessing

### Claude Models (Anthropic)
- Claude-3-opus: Uses messages.create API with "claude-3-opus-20240229"
- Claude-3.7-sonnet: Uses messages.create API with "claude-3-7-sonnet-20250219"
- Accepts base64 encoded images in message content
- Supports system prompts and 300 token limit

## Common Features
- All models support:
  - System prompts for instruction
  - Image + text input processing
  - Exception handling with "E) Error" fallback
  - Seed setting for reproducibility
  - Multiple choice question format standardization

## Random Seed Checklist
- [x] gpt-4o
- [x] gpt-4o-mini
- [ ] gemini-1.5-pro
- [ ] gemini-2.0-flash
- [x] claude-3.7-sonnet (not supported)
- [x] qwen-vl-max
- [x] qwen-vl-plus
- [ ] qwen-2.5-7b
- [ ] qwen-2.5-32b
- [ ] llama-3.2-11b
- [ ] llama-3.2-90b
- [x] llava-1.5
- [x] llava-1.6
- [x] o4-mini (not supported in Response API)
- [] o3 (not supported in Response API)