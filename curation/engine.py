import os
from PIL import Image
import textwrap

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    AutoProcessor,
    LlavaForConditionalGeneration,
)
from openai import OpenAI
import anthropic
import torch

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import SUPPORTED_BENCHMARKS, SUPPORTED_MODELS
from utils.data_preparation import (
    get_sample_by_image_id,
    get_image_path_by_id,
    encode_image,
)


class VLMInterface:
    def __init__(self, model_type):
        assert (
            model_type in SUPPORTED_MODELS
        ), f"Unsupported model: {model_type}; supported models are {SUPPORTED_MODELS}"
        # default system instruction
        self._system_prompt = "You will only answer multiple choice questions with single answers as the options (A), (B), (C), or (D). You will answer them correctly. You will not use any fullstops or punctuation. You will not explain your answers or write words before or after the answers. Only the answer itself will you respond with."
        self.model_type = model_type
        self.init_model()

    @property
    def system_prompt(self):
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, system_prompt):
        self._system_prompt = system_prompt

    def init_model(self):
        model_prefix = self.model_type.split("-")[0]
        if model_prefix == "gemini":
            genai.configure(api_key=os.environ["GEMINI_KEY"])
            generation_config = {
                "temperature": 0,
                "top_p": 1,
                "max_output_tokens": 300,
                "response_mime_type": "text/plain",
            }
            self.model = genai.GenerativeModel(
                model_name=self.model_type,
                generation_config=generation_config,
                system_instruction=self.system_prompt,
            )
        elif self.model_type == "llava-1.6":
            self.processor = LlavaNextProcessor.from_pretrained(
                "llava-hf/llava-v1.6-mistral-7b-hf"
            )
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                "llava-hf/llava-v1.6-mistral-7b-hf",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                load_in_4bit=True,
                cache_dir=os.getenv("MODEL_CACHE_DIR"),
            )
        elif self.model_type == "llava-1.5":
            self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                load_in_4bit=True,
                cache_dir=os.getenv("MODEL_CACHE_DIR"),
            )
        elif model_prefix == "llama":
            if self.model_type == "llama-3.2-11b":
                model_name = "unsloth/Llama-3.2-11B-Vision-Instruct"
            elif self.model_type == "llama-3.2-90b":
                model_name = "unsloth/Llama-3.2-90B-Vision-Instruct"
            else:
                raise ValueError(
                    f"Unsupported model: {self.model_type}; supported models are {SUPPORTED_MODELS}"
                )

            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                model_name,
                # Use 4bit to reduce memory use. False for 16bit LoRA.
                load_in_4bit=True,
                use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
                cache_dir=os.getenv("MODEL_CACHE_DIR"),
            )
            FastVisionModel.for_inference(self.model)  # Enable for inference!
        elif (
            self.model_type == "gpt-4o"
            or self.model_type == "gpt-4o-mini"
            or self.model_type == "o4-mini"
            or self.model_type == "o3"
        ):
            self.model = OpenAI(api_key=os.getenv("OPENAI_KEY"))
        elif model_prefix == "qwen":
            if self.model_type == "qwen-vl-max" or self.model_type == "qwen-vl-plus":
                self.model = OpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                )
            elif (
                self.model_type == "qwen-2.5-7b"
                or self.model_type == "qwen-2.5-32b"
                or self.model_type == "qwen-2.5-72b"
            ):
                if self.model_type == "qwen-2.5-7b":
                    model_name = "unsloth/Qwen2.5-VL-7B-Instruct"
                elif self.model_type == "qwen-2.5-32b":
                    model_name = "unsloth/Qwen2.5-VL-32B-Instruct"
                elif self.model_type == "qwen-2.5-72b":
                    model_name = "unsloth/Qwen2.5-VL-72B-Instruct"
                else:
                    raise ValueError(
                        f"Unsupported model: {self.model_type}; supported models are {SUPPORTED_MODELS}"
                    )

                self.model, self.tokenizer = FastVisionModel.from_pretrained(
                    model_name,
                    load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
                    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
                    cache_dir=os.getenv("MODEL_CACHE_DIR"),
                )
                FastVisionModel.for_inference(self.model)  # Enable for inference!
            else:
                raise ValueError(
                    f"Unsupported model: {self.model_type}; supported models are {SUPPORTED_MODELS}"
                )
        elif (
            self.model_type == "claude-3-opus" or self.model_type == "claude-3.7-sonnet"
        ):
            self.model = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        else:
            raise ValueError(
                f"Unsupported model: {self.model_type}; supported models are {SUPPORTED_MODELS}"
            )

    def get_response(self, prompt, image_path):
        image = Image.open(image_path)
        model_prefix = self.model_type.split("-")[0]
        device = torch.cuda.current_device()

        if model_prefix == "gemini":
            try:
                response = self.model.generate_content(
                    [prompt, image],
                    config=GenerationConfig(
                        system_instruction=self.system_prompt,
                    ),
                )
                return response.text
            except Exception as e:
                print(f"An error occurred: {e}")
                return "E) Error"
        elif model_prefix == "llava":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )

            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
                device
            )
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                return_dict_in_generate=True,
                output_scores=True,
            )
            decoded = self.processor.decode(
                output.sequences[0], skip_special_tokens=True
            )
            return decoded
        elif model_prefix == "llama":
            messages = [
                {
                    "role": "user",  #  Prompting with images is incompatible with system messages.
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                },
            ]
            input_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                image, input_text, add_special_tokens=False, return_tensors="pt"
            ).to(device)
            output = self.model.generate(
                **inputs, max_new_tokens=300, use_cache=True, temperature=1.5, min_p=0.1
            )
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return decoded

        elif model_prefix == "gpt":
            if self.model_type == "gpt-4o":
                model_name = "gpt-4o-2024-08-06"
            elif self.model_type == "gpt-4o-mini":
                model_name = "gpt-4o-mini-2024-07-18"
            else:
                raise ValueError(
                    f"Unsupported model: {self.model_type}; supported models are {SUPPORTED_MODELS}"
                )

            base64_image = encode_image(image_path)
            context = [
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "text",
                            "text": self.system_prompt,
                        }
                    ],
                }
            ]
            context.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            )
            try:
                response = self.model.chat.completions.create(
                    model=model_name,
                    messages=context,
                    max_tokens=300,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"An error occurred: {e}")
                return "E) Error"

        elif self.model_type == "o4-mini":
            base64_image = encode_image(image_path)
            context = [
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self.system_prompt,
                        }
                    ],
                }
            ]
            context.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            )
            try:
                response = self.model.responses.create(
                    model="o4-mini",
                    reasoning={"effort": "medium"},
                    input=context,
                    max_output_tokens=3000,
                )

                if (
                    response.status == "incomplete"
                    and response.incomplete_details.reason == "max_output_tokens"
                ):
                    print("Ran out of tokens")
                    if response.output_text:
                        print("Partial output:", response.output_text)
                    else:
                        print("Ran out of tokens during reasoning")

                return response.output_text

            except Exception as e:
                print(f"An error occurred: {e}")
                return "E) Error"

        elif self.model_type == "o3":
            base64_image = encode_image(image_path)
            context = [
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self.system_prompt,
                        }
                    ],
                }
            ]
            context.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            )
            try:
                response = self.model.responses.create(
                    model="o3",
                    reasoning={"effort": "medium"},
                    input=context,
                    max_output_tokens=3000,
                )

                if (
                    response.status == "incomplete"
                    and response.incomplete_details.reason == "max_output_tokens"
                ):
                    print("Ran out of tokens")
                    if response.output_text:
                        print("Partial output:", response.output_text)
                    else:
                        print("Ran out of tokens during reasoning")

                return response.output_text

            except Exception as e:
                print(f"An error occurred: {e}")
                return "E) Error"

        elif (
            self.model_type == "qwen-2.5-7b"
            or self.model_type == "qwen-2.5-32b"
            or self.model_type == "qwen-2.5-72b"
        ):
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                },
            ]
            input_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                image, input_text, add_special_tokens=False, return_tensors="pt"
            ).to(device)
            output = self.model.generate(
                **inputs, max_new_tokens=300, use_cache=True, temperature=1.5, min_p=0.1
            )
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return decoded

        elif model_prefix == "qwen":
            base64_image = encode_image(image_path)
            context = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.system_prompt,
                        }
                    ],
                }
            ]
            context.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            )
            try:
                response = self.model.chat.completions.create(
                    model=self.model_type,
                    messages=context,
                    max_tokens=300,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"An error occurred: {e}")
                return "E) Error"
        elif model_prefix == "claude":
            if self.model_type == "claude-3-opus":
                model_name = "claude-3-opus-20240229"
            elif self.model_type == "claude-3.7-sonnet":
                model_name = "claude-3-7-sonnet-20250219"
            else:
                raise ValueError(
                    f"Unsupported model: {self.model_type}; supported models are {SUPPORTED_MODELS}"
                )

            base64_image = encode_image(image_path)

            image_type = image_path.split(".")[-1]
            if image_type == "jpg":
                image_type = "jpeg"

            context = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{image_type}",
                                "data": base64_image,
                            },
                        },
                    ],
                },
            ]
            try:
                message = self.model.messages.create(
                    model=model_name,
                    max_tokens=300,
                    system=self.system_prompt,
                    messages=context,
                )
                print(message.content[0].text)
                return message.content[0].text
            except Exception as e:
                print(f"An error occurred: {e}")
                return "E) Error"
        else:
            raise ValueError(
                f"Unsupported model: {self.model_type}; supported models are {SUPPORTED_MODELS}"
            )


def prompt_adjustment(model, prompt_strategy, question):
    if prompt_strategy == "chain_of_thought":
        model.system_prompt = textwrap.dedent(
            """
                You will be given an image, and a multiple choice question regarding the image. Describe the image in detail first, then answer based on your description. Think step by step and give a final answer. You will include one of the choices (A), (B), (C), or (D) in your final answer. 
            """
        )
        user_prompt = question
    elif prompt_strategy == "spurious_aware":
        model.system_prompt = textwrap.dedent(
            """
                You will be given an image, and a multiple choice question regarding the image. Be aware that there may be some spurious features in the image that associate with some of the options. Describe the potential spurious features. Then give a answer without using the spurious features. You will include one of the choices (A), (B), (C), or (D) in your final answer.  
            """
        )
        user_prompt = question
    else:
        model.system_prompt = textwrap.dedent(
            """
                You will be given an image, and a multiple choice question regarding the image. You will provide your answer as one of the options (A), (B), (C), or (D). You will answer correctly. You will not use any fullstops or punctuation. You will not explain your answer or write words before or after the answer. Only the answer itself will you respond with.
            """
        )
        user_prompt = question
    return user_prompt


def get_prediction(
    image_id: str,
    model: VLMInterface,
    benchmark: str,
    prompt_strategy: str,
    image_path: str = None,
):
    # Assertions for models and benchmarks
    assert (
        benchmark in SUPPORTED_BENCHMARKS
    ), f"Unsupported benchmark: {benchmark}; supported benchmarks are {SUPPORTED_BENCHMARKS}"
    # Format question
    # Get question by image_id
    sample = get_sample_by_image_id(image_id, benchmark)
    question = f"""
        Question: {sample['question']}
        Options:
        (A) {sample['A']}
        (B) {sample['B']}
        (C) {sample['C']}
        (D) {sample['D']}
        Please select the correct answer from the options above.
    """
    question = textwrap.dedent(question)

    # Prepare image
    if image_path is None:
        image_path = get_image_path_by_id(image_id, benchmark)
    # Prompt strategy adjustment
    user_prompt = prompt_adjustment(model, prompt_strategy, question)

    prediction = model.get_response(user_prompt, image_path)
    return prediction
