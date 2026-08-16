import warnings
import torch
from torch.utils.data import Dataset
from PIL import Image

# Some HAM10000 JPEGs have malformed EXIF metadata; PIL prints a UserWarning
# to stderr per image on load, which breaks tqdm's in-place progress bar.
warnings.filterwarnings("ignore", message="Corrupt EXIF data*", category=UserWarning)


class HAM10000Dataset(Dataset):
    def __init__(self, df, transform, metadata_df=None):
        """
        metadata_df: optional dataframe indexed by image_id (see
        src.data.metadata.load_metadata_features) holding the per-image
        metadata feature vector. When provided, __getitem__ returns
        (img, metadata, label) instead of (img, label) — metadata is looked
        up by each row's image_id, not by position, so it stays correctly
        paired with its image even if the two dataframes are ordered
        differently or one contains extra rows.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.metadata_df = metadata_df

        if self.metadata_df is not None:
            missing_ids = set(self.df["image_id"]) - set(self.metadata_df.index)
            if missing_ids:
                raise ValueError(
                    f"{len(missing_ids)} image_id(s) in df have no matching metadata row, "
                    f"e.g. {list(missing_ids)[:5]}"
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        img = self.transform(img)
        label = int(row["label"])

        if self.metadata_df is None:
            return img, label

        metadata = torch.tensor(
            self.metadata_df.loc[row["image_id"]].values, dtype=torch.float32
        )
        return img, metadata, label