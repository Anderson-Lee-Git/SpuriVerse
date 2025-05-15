import os
import shutil

import pandas as pd
import requests
import json
from dotenv import load_dotenv
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_preparation import (
    get_image_path_by_id,
    encode_image,
    get_sample_by_image_id,
)
from utils.constants import SUPPORTED_BENCHMARKS
from curation.engine import VLMInterface, get_prediction

load_dotenv()

logger = logging.getLogger(__name__)


class Pipeline:
    @staticmethod
    def send_generation_request(
        host,
        params,
    ):
        headers = {
            "Accept": "image/*",
            "Authorization": f"Bearer {os.getenv('STABILITY_KEY')}",
        }

        # Encode parameters
        files = {}
        image = params.pop("image", None)
        mask = params.pop("mask", None)
        if image is not None and image != "":
            files["image"] = open(image, "rb")
        if mask is not None and mask != "":
            files["mask"] = open(mask, "rb")
        if len(files) == 0:
            files["none"] = ""

        # Send request
        print(f"Sending REST request to {host}...")
        response = requests.post(host, headers=headers, files=files, data=params)
        if not response.ok:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        return response

    @staticmethod
    def generate_images(description, directory, filename, num_imgs=25):
        # make directory and clear if it exists
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

        for i in range(num_imgs):
            # @title Stable Image Ultra
            prompt = description  # @param {type:"string"}
            negative_prompt = ""  # @param {type:"string"}
            aspect_ratio = "3:2"  # @param ["21:9", "16:9", "3:2", "5:4", "1:1", "4:5", "2:3", "9:16", "9:21"]
            seed = i + 1  # @param {type:"integer"}
            output_format = "png"  # @param ["webp", "jpeg", "png"]

            host = "https://api.stability.ai/v2beta/stable-image/generate/ultra"

            params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "aspect_ratio": aspect_ratio,
                "seed": seed,
                "output_format": output_format,
            }

            response = Pipeline.send_generation_request(host, params)

            # Decode response
            output_image = response.content
            finish_reason = response.headers.get("finish-reason")
            seed = response.headers.get("seed")

            # Check for NSFW classification
            if finish_reason == "CONTENT_FILTERED":
                raise Warning("Generation failed NSFW classifier")

            # Save and display result
            generated = os.path.join(directory, f"{filename}_{seed}.{output_format}")
            with open(generated, "wb") as f:
                f.write(output_image)
            print(f"Saved image {generated}")

    @staticmethod
    def evaluate(
        image_id,
        directory,
        model: VLMInterface,
        model_type,
        benchmark,
        prompt_strategy,
    ):
        image_ids = []
        responses = []
        if type(directory) != str:
            directory = str(directory)
        for f in os.listdir(directory):
            if not f.endswith(".png"):
                continue
            logger.info(f"Processing {f} in {directory}")
            image_path = os.path.join(directory, f)
            responses.append(
                get_prediction(
                    image_id=image_id,
                    model=model,
                    benchmark=benchmark,
                    prompt_strategy=prompt_strategy,
                    image_path=image_path,
                )
            )
            image_ids.append(f)

        data = {
            "image_id": image_ids,
            "response": responses,
        }
        df = pd.DataFrame(data)

        directory_name = os.path.join(
            os.getenv("GROUP_RESPONSES_PATH"), benchmark, model_type, prompt_strategy
        )

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        df.to_csv(
            os.path.join(
                directory_name,
                f'group_responses_{directory.split("/")[-1]}.csv',
            ),
            index=False,
        )
        return df

    @staticmethod
    def extract_spurious_attributes(image_id, client, benchmark):
        """
        1. Extract top 2 core and spurious attributes from the image (context: image, question)
        """
        sample = get_sample_by_image_id(image_id, benchmark)
        image_path = get_image_path_by_id(image_id, benchmark)
        base64_image = encode_image(image_path)
        img_A = sample["A"]
        img_B = sample["B"]
        img_C = sample["C"]
        img_D = sample["D"]
        prediction = sample["parsed"]
        answer = sample["answer"]

        question = f"""
        {sample['question']}
        (A) {img_A}
        (B) {img_B}
        (C) {img_C}
        (D) {img_D}
        """
        print(question)

        context = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant to determine if a model's error is caused primarily by spurious correlations, patterns that can often be used to predict the target, but are not actually causal.",
                    }
                ],
            },
        ]
        context.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Given this image, a Large Multi-modal Model was asked, {sample['question']}, and given the choices: A) {img_A} B) {img_B} C) {img_C} D) {img_D}.  The model chose {prediction} and the correct answer is {answer}. The error is most likely due to spurious correlation. List the top two spurious attributes that the model may have used to predict the wrong answer {prediction}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=context,
                max_tokens=500,
                temperature=0,
                top_p=1,
            )
            response_content = response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
        print(response_content)
        return response_content

    @staticmethod
    def generate_descriptions(client, image_id, attributes, benchmark):
        sample = get_sample_by_image_id(image_id, benchmark)
        question = sample["question"]
        answer = sample["answer"]

        default_prompt = f"""
        You are given the following:
        question: {question}
        answer: {answer}
        spurious attribute: {attributes}

        Based on the question and the answer, generate a description of a scene such that when the question is asked, the answer is {answer}. Keep the description to one short sentence.

        Write another one sentence description that includes the spurious attribute while maintaining the same context.

        Return the response in JSON format with the two keys: "positive" and "negative" where "positive" describes the scene with the spurious attribute and "negative" describes the scene without the spurious attribute.
        """
        context = []
        context.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": default_prompt,
                    },
                ],
            }
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=context,
                max_tokens=500,
                temperature=0,
                top_p=1,
                response_format={"type": "json_object"},
            )

            response_content = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"An error occurred: {e}")

        return (
            response_content["positive"],
            response_content["negative"],
        )

    @staticmethod
    def retrieve_spurious_attributes(image_id, benchmark):
        assert benchmark in SUPPORTED_BENCHMARKS, f"Benchmark {benchmark} not supported"
        sample = get_sample_by_image_id(image_id, benchmark)
        assert (
            "spuriousAttr" in sample.keys()
        ), "spuriousAttr key must be present in the sample"
        return sample["spuriousAttr"]

    @staticmethod
    def save_results(results, save_dir):
        if not pd.api.types.is_string_dtype(results["sample_id"]):
            results["sample_id"] = str(results["sample_id"])
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, "spurious_correlation_candidates.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if not pd.api.types.is_string_dtype(df["sample_id"]):
                df["sample_id"] = df["sample_id"].astype(str)
            new_row = pd.DataFrame([results])
            # Concatenate the new row to the existing DataFrame
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(file_path, index=False)
            print("CSV file saved successfully.")
        else:
            df = pd.DataFrame([results])
            # Save the DataFrame to a CSV file
            df.to_csv(file_path, index=False)
            print("CSV file saved successfully.")
