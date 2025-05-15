import os
import random

import streamlit as st
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from curation.pipeline import Pipeline
from curation.engine import VLMInterface
from utils.data_preparation import (
    get_sample_by_image_id,
    get_image_path_by_id,
    get_is_filtered_benchmark_id_set,
)
from utils.constants import SUPPORTED_BENCHMARKS

load_dotenv(".env")

BENCHMARK_OPTIONS = SUPPORTED_BENCHMARKS


def pipeline_page():
    """
    Redesigned Pipeline Page with two-column layout and improved flow control.
    Left column: Benchmark selection and sample navigation.
    Right column: Pipeline steps and processing flow.
    """
    st.set_page_config(layout="wide")
    # Initialize session state variables if not already set
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1  # Track the current active pipeline step

    if "benchmark" not in st.session_state:
        st.session_state.benchmark = BENCHMARK_OPTIONS[0]

    if "sample_id" not in st.session_state:
        st.session_state.sample_id = None

    if "sample_index" not in st.session_state:
        st.session_state.sample_index = 0

    if "benchmark_samples" not in st.session_state:
        st.session_state.benchmark_samples = []

    if "extracted_attributes" not in st.session_state:
        st.session_state.extracted_attributes = None

    if "final_attributes" not in st.session_state:
        st.session_state.final_attributes = None

    if "pos_description" not in st.session_state:
        st.session_state.pos_description = None

    if "neg_description" not in st.session_state:
        st.session_state.neg_description = None

    if "final_pos_description" not in st.session_state:
        st.session_state.final_pos_description = None

    if "final_neg_description" not in st.session_state:
        st.session_state.final_neg_description = None

    if "filename" not in st.session_state:
        st.session_state.filename = None

    # Set up pipeline and client
    pipeline = Pipeline()
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

    # Main layout with two columns
    left_col, right_col = st.columns([1, 2])

    # --- LEFT COLUMN: BENCHMARK AND SAMPLE NAVIGATION ---
    with left_col:
        st.header("Sample Navigation")

        # Benchmark Selection
        with st.container(border=True):
            st.subheader("Benchmark Selection")

            # Update benchmark and sample list when selection changes
            def on_benchmark_change():
                benchmark = st.session_state.benchmark_select
                st.session_state.benchmark = benchmark
                # Get samples for this benchmark
                sample_ids = list(get_is_filtered_benchmark_id_set(benchmark))
                sample_ids.sort()  # Sort for consistent navigation
                st.session_state.benchmark_samples = sample_ids
                # Reset sample index
                st.session_state.sample_index = 0
                if sample_ids:
                    st.session_state.sample_id = sample_ids[0]
                    # Reset all processing state
                    reset_processing_state()

            st.selectbox(
                "Select Benchmark",
                options=BENCHMARK_OPTIONS,
                key="benchmark_select",
                on_change=on_benchmark_change,
                index=BENCHMARK_OPTIONS.index(st.session_state.benchmark),
            )

        # Sample Navigation Controls
        with st.container(border=True):
            st.subheader("Sample Navigation")

            # Load or refresh benchmark samples if needed
            if not st.session_state.benchmark_samples:
                sample_ids = list(
                    get_is_filtered_benchmark_id_set(st.session_state.benchmark)
                )
                sample_ids.sort()
                st.session_state.benchmark_samples = sample_ids
                if sample_ids:
                    st.session_state.sample_id = sample_ids[0]

            total_samples = len(st.session_state.benchmark_samples)

            # Sample navigation functions
            def go_to_prev_sample():
                if st.session_state.sample_index > 0:
                    st.session_state.sample_index -= 1
                    st.session_state.sample_id = st.session_state.benchmark_samples[
                        st.session_state.sample_index
                    ]
                    reset_processing_state()

            def go_to_next_sample():
                if st.session_state.sample_index < total_samples - 1:
                    st.session_state.sample_index += 1
                    st.session_state.sample_id = st.session_state.benchmark_samples[
                        st.session_state.sample_index
                    ]
                    reset_processing_state()

            def go_to_random_sample():
                if total_samples > 0:
                    st.session_state.sample_index = random.randint(0, total_samples - 1)
                    st.session_state.sample_id = st.session_state.benchmark_samples[
                        st.session_state.sample_index
                    ]
                    reset_processing_state()

            def reset_processing_state():
                """Reset all processing state when changing samples"""
                st.session_state.current_step = 1
                st.session_state.extracted_attributes = None
                st.session_state.final_attributes = None
                st.session_state.pos_description = None
                st.session_state.neg_description = None
                st.session_state.final_pos_description = None
                st.session_state.final_neg_description = None
                st.session_state.filename = None

            # Navigation buttons
            col1, col2, col3 = st.columns(3)
            col1.button(
                "◀ Previous",
                on_click=go_to_prev_sample,
                disabled=total_samples == 0 or st.session_state.sample_index == 0,
            )
            col2.button(
                "🎲 Random", on_click=go_to_random_sample, disabled=total_samples == 0
            )
            col3.button(
                "Next ▶",
                on_click=go_to_next_sample,
                disabled=total_samples == 0
                or st.session_state.sample_index == total_samples - 1,
            )

            # Sample progress indicator
            if total_samples > 0:
                st.progress(
                    st.session_state.sample_index / (total_samples - 1)
                    if total_samples > 1
                    else 1.0
                )
                st.text(
                    f"Sample {st.session_state.sample_index + 1} of {total_samples}"
                )
                st.text(f"Sample ID: {st.session_state.sample_id}")

    # --- RIGHT COLUMN: PIPELINE STEPS ---
    with right_col:
        st.header("Current Sample")
        # Current Sample Display
        if st.session_state.sample_id:
            with st.container(border=True):
                try:
                    sample = get_sample_by_image_id(
                        st.session_state.sample_id, st.session_state.benchmark
                    )
                    image_path = get_image_path_by_id(
                        st.session_state.sample_id, st.session_state.benchmark
                    )
                    st.image(image_path)
                    st.markdown("**Question:**")
                    st.markdown(sample["question"])
                    st.markdown("**Options:**")
                    st.markdown(f"(A) {sample['A']}")
                    st.markdown(f"(B) {sample['B']}")
                    st.markdown(f"(C) {sample['C']}")
                    st.markdown(f"(D) {sample['D']}")
                    st.markdown(f"**Prediction:** {sample['parsed']}")
                    st.markdown(f"**Answer:** {sample['answer']}")
                except Exception as e:
                    st.error(f"Error loading sample: {e}")
        st.header("Pipeline Steps")

        # Custom expander with active/complete status
        def step_container(step_num, title, is_active=False, is_complete=False):
            with st.container(border=True):
                # Status indicator
                status = (
                    "🔄 Active"
                    if is_active
                    else "✅ Complete" if is_complete else "⏳ Pending"
                )
                st.markdown(f"### Step {step_num}: {title} ({status})")
                return st.container()

        # Function to check if each step is active or complete
        def is_step_active(step_num):
            return st.session_state.current_step == step_num

        def is_step_complete(step_num):
            return st.session_state.current_step > step_num

        def advance_to_step(step_num):
            st.session_state.current_step = max(st.session_state.current_step, step_num)

        # Only show pipeline if a sample is selected
        if st.session_state.sample_id:
            # Step 1: Extract Spurious Attributes
            with step_container(
                1,
                "Extract Spurious Attributes",
                is_active=is_step_active(1),
                is_complete=is_step_complete(1),
            ):
                if is_step_active(1) or is_step_complete(1):
                    col1, col2 = st.columns([3, 1])

                    # Run extraction
                    if col1.button("Extract Attributes", disabled=is_step_complete(1)):
                        with st.spinner("Extracting spurious attributes..."):
                            extracted_attributes = pipeline.extract_spurious_attributes(
                                st.session_state.sample_id,
                                client,
                                st.session_state.benchmark,
                            )
                            st.session_state.extracted_attributes = extracted_attributes
                            st.session_state.final_attributes = extracted_attributes
                            advance_to_step(2)
                            st.rerun()

                    # Load from stored
                    if col2.button("From Stored", disabled=is_step_complete(1)):
                        stored_attributes = pipeline.retrieve_spurious_attributes(
                            st.session_state.sample_id, st.session_state.benchmark
                        )
                        st.session_state.extracted_attributes = stored_attributes
                        st.session_state.final_attributes = stored_attributes
                        advance_to_step(2)
                        st.rerun()

                # Display and edit attributes
                if is_step_complete(1):
                    st.text_area(
                        "Spurious Attributes",
                        value=st.session_state.extracted_attributes,
                        height=150,
                        key="final_attributes",
                        disabled=not is_step_active(2),
                    )

            # Step 2: Generate Image Descriptions
            with step_container(
                2,
                "Generate Image Descriptions",
                is_active=is_step_active(2),
                is_complete=is_step_complete(2),
            ):
                if is_step_active(2) or is_step_complete(2):
                    if st.button("Generate Descriptions", disabled=is_step_complete(2)):
                        with st.spinner("Generating image descriptions..."):
                            pos_description, neg_description = (
                                pipeline.generate_descriptions(
                                    client,
                                    st.session_state.sample_id,
                                    st.session_state.final_attributes,
                                    st.session_state.benchmark,
                                )
                            )
                            st.session_state.pos_description = pos_description
                            st.session_state.neg_description = neg_description
                            st.session_state.final_pos_description = pos_description
                            st.session_state.final_neg_description = neg_description
                            advance_to_step(3)
                            st.rerun()

                # Display and edit descriptions
                if is_step_complete(2):
                    st.text_area(
                        "Positive Description",
                        value=st.session_state.pos_description,
                        height=150,
                        key="final_pos_description",
                        disabled=not is_step_active(3),
                    )
                    st.text_area(
                        "Negative Description",
                        value=st.session_state.neg_description,
                        height=150,
                        key="final_neg_description",
                        disabled=not is_step_active(3),
                    )

            # Step 3: Generate Images
            with step_container(
                3,
                "Generate Images",
                is_active=is_step_active(3),
                is_complete=is_step_complete(3),
            ):
                if is_step_active(3) or is_step_complete(3):
                    if st.button("Generate Images", disabled=is_step_complete(3)):
                        with st.spinner("Generating images..."):
                            filename = st.session_state.sample_id
                            pos_directory = os.path.join(
                                os.getenv("GROUP_GENERATION_EXPERIMENT_PATH"),
                                st.session_state.benchmark,
                                filename,
                            )
                            neg_directory = os.path.join(
                                os.getenv("GROUP_GENERATION_EXPERIMENT_PATH"),
                                st.session_state.benchmark,
                                f"{filename}_counter",
                            )
                            num_imgs = 1  # Number of images to generate

                            # Generate positive images
                            pipeline.generate_images(
                                description=st.session_state.final_pos_description,
                                directory=pos_directory,
                                filename=filename,
                                num_imgs=num_imgs,
                            )

                            # Generate negative images
                            pipeline.generate_images(
                                description=st.session_state.final_neg_description,
                                directory=neg_directory,
                                filename=f"{filename}_counter",
                                num_imgs=num_imgs,
                            )
                            advance_to_step(4)
                            st.rerun()

                # Display generated images
                if is_step_complete(3):
                    col1, col2 = st.columns(2)
                    col1.markdown("**Positive Images**")
                    col2.markdown("**Negative Images**")

                    filename = st.session_state.sample_id
                    pos_directory = os.path.join(
                        os.getenv("GROUP_GENERATION_EXPERIMENT_PATH"),
                        st.session_state.benchmark,
                        filename,
                    )
                    neg_directory = os.path.join(
                        os.getenv("GROUP_GENERATION_EXPERIMENT_PATH"),
                        st.session_state.benchmark,
                        f"{filename}_counter",
                    )

                    if os.path.exists(pos_directory) and os.path.exists(neg_directory):
                        for f in os.listdir(pos_directory):
                            if f.endswith(".png"):
                                col1.image(os.path.join(pos_directory, f))
                        for f in os.listdir(neg_directory):
                            if f.endswith(".png"):
                                col2.image(os.path.join(neg_directory, f))

            # Step 4: Evaluate Generated Images
            with step_container(
                4,
                "Evaluate Generated Images",
                is_active=is_step_active(4),
                is_complete=is_step_complete(4),
            ):
                if is_step_active(4) or is_step_complete(4):
                    if st.button("Evaluate Images", disabled=is_step_complete(4)):
                        with st.spinner("Evaluating images..."):
                            filename = st.session_state.sample_id
                            pos_directory = os.path.join(
                                os.getenv("GROUP_GENERATION_EXPERIMENT_PATH"),
                                st.session_state.benchmark,
                                filename,
                            )
                            neg_directory = os.path.join(
                                os.getenv("GROUP_GENERATION_EXPERIMENT_PATH"),
                                st.session_state.benchmark,
                                f"{filename}_counter",
                            )
                            model = VLMInterface(model_type="gpt-4o")

                            pipeline.evaluate(
                                image_id=st.session_state.sample_id,
                                directory=pos_directory,
                                model=model,
                                model_type="gpt-4o",
                                benchmark=st.session_state.benchmark,
                                prompt_strategy="direct_prompting",
                            )

                            pipeline.evaluate(
                                image_id=st.session_state.sample_id,
                                directory=neg_directory,
                                model=model,
                                model_type="gpt-4o",
                                benchmark=st.session_state.benchmark,
                                prompt_strategy="direct_prompting",
                            )
                            advance_to_step(5)
                            st.rerun()

                # Display evaluation results
                if is_step_complete(4):
                    col1, col2 = st.columns(2)
                    col1.markdown("**Positive Group Evaluation**")
                    col2.markdown("**Negative Group Evaluation**")

                    filename = st.session_state.sample_id
                    pos_evaluation_path = os.path.join(
                        os.getenv("GROUP_RESPONSES_PATH"),
                        st.session_state.benchmark,
                        "gpt-4o",
                        "direct_prompting",
                        f"group_responses_{filename}.csv",
                    )
                    neg_evaluation_path = os.path.join(
                        os.getenv("GROUP_RESPONSES_PATH"),
                        st.session_state.benchmark,
                        "gpt-4o",
                        "direct_prompting",
                        f"group_responses_{filename}_counter.csv",
                    )

                    if os.path.exists(pos_evaluation_path) and os.path.exists(
                        neg_evaluation_path
                    ):
                        col1.table(pd.read_csv(pos_evaluation_path))
                        col2.table(pd.read_csv(neg_evaluation_path))

            # Step 5: Save Results
            with step_container(
                5,
                "Save Results",
                is_active=is_step_active(5),
                is_complete=is_step_complete(5),
            ):
                if is_step_active(5):
                    filename_input = st.text_input(
                        "Filename for results (leave empty to use sample ID)",
                        key="filename_input",
                        value=st.session_state.sample_id,
                    )

                    if st.button("Save Results"):
                        with st.spinner("Saving results..."):
                            if not filename_input:
                                st.session_state.filename = st.session_state.sample_id
                            else:
                                st.session_state.filename = filename_input

                            src_pos_directory = os.path.join(
                                os.getenv("GROUP_GENERATION_EXPERIMENT_PATH"),
                                st.session_state.benchmark,
                                f"{st.session_state.sample_id}/*",
                            )
                            src_neg_directory = os.path.join(
                                os.getenv("GROUP_GENERATION_EXPERIMENT_PATH"),
                                st.session_state.benchmark,
                                f"{st.session_state.sample_id}_counter/*",
                            )
                            dst_pos_directory = os.path.join(
                                os.getenv("GROUP_GENERATION_PROD_PATH"),
                                st.session_state.benchmark,
                                f"{st.session_state.sample_id}",
                            )
                            dst_neg_directory = os.path.join(
                                os.getenv("GROUP_GENERATION_PROD_PATH"),
                                st.session_state.benchmark,
                                f"{st.session_state.sample_id}_counter",
                            )
                            os.makedirs(dst_pos_directory, exist_ok=True)
                            os.makedirs(dst_neg_directory, exist_ok=True)

                            os.system(f"cp -rf {src_pos_directory} {dst_pos_directory}")
                            os.system(f"cp -rf {src_neg_directory} {dst_neg_directory}")

                            results = {
                                "sample_id": st.session_state.sample_id,
                                "extracted_attributes": st.session_state.extracted_attributes,
                                "final_attributes": st.session_state.final_attributes,
                                "pos_description": st.session_state.pos_description,
                                "neg_description": st.session_state.neg_description,
                                "final_pos_description": st.session_state.final_pos_description,
                                "final_neg_description": st.session_state.final_neg_description,
                            }

                            save_dir = os.path.join(
                                os.getenv("HUMAN_ACCEPTED_DIR"),
                                st.session_state.benchmark,
                            )
                            pipeline.save_results(results, save_dir)
                            st.success("Results saved successfully!")
                            advance_to_step(6)  # Mark complete

                if is_step_complete(5):
                    st.success(f"Results saved for sample {st.session_state.sample_id}")


if __name__ == "__main__":
    pipeline_page()
