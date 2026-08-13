import warnings
from torch.utils.data import Dataset
from PIL import Image

# Some HAM10000 JPEGs have malformed EXIF metadata; PIL prints a UserWarning
# to stderr per image on load, which breaks tqdm's in-place progress bar.
warnings.filterwarnings("ignore", message="Corrupt EXIF data*", category=UserWarning)


class HAM10000Dataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        img = self.transform(img)
        label = int(row["label"])
        return img, label