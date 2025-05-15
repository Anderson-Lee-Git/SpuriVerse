import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from PIL import Image
import io
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_preparation import get_image_path_by_id, convert_to_str

load_dotenv()


ds = load_dataset("yanyiwei/SpuriVerse")
df = ds["train"].to_pandas()
# save anchor set ids and benchmark
print(
    f"Save reference anchor set to {Path(os.getenv('EVAL_HUMAN_ACCEPTED_DIR')) / 'reference_anchor_set.csv'}"
)
df["sample_ids"] = df["image_id"].apply(convert_to_str)
reference_anchor_df = df.loc[:, ["sample_ids", "benchmark"]]
reference_anchor_df.to_csv(
    Path(os.getenv("EVAL_HUMAN_ACCEPTED_DIR")) / "reference_anchor_set.csv",
    index=False,
)
# save anchor set images if not exist
print(f"Save anchor set images if not exist")
pbar = tqdm(total=len(reference_anchor_df))
for index, row in reference_anchor_df.iterrows():
    image_path = get_image_path_by_id(row["sample_ids"], row["benchmark"])
    if not os.path.exists(image_path):
        print(f"Image {row['sample_ids']} does not exist")
        image = row["image"]
        image = Image.open(io.BytesIO(image))
        image.save(image_path)
    pbar.update(1)
pbar.close()
# save spurious group images for finetuning
print(
    f"Save spurious group images for finetuning to {Path(os.getenv('GROUP_GENERATION_PROD_PATH'))}"
)
base_dir = Path(os.getenv("GROUP_GENERATION_PROD_PATH"))
pbar = tqdm(total=len(df))
for index, row in df.iterrows():
    group_dir = base_dir / row["benchmark"]
    image_dir = group_dir / f"{row['sample_ids']}"
    image_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 11):
        image = row[f"spurious_group_{i}"]["bytes"]
        image = Image.open(io.BytesIO(image))
        image.save(image_dir / f"{row['sample_ids']}_{i}.png")
    pbar.update(1)
pbar.close()
