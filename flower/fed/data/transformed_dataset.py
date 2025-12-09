from PIL import Image
from torch.utils.data import Dataset


class TransformedDataset(Dataset):
  def __init__(self, hf_dataset, transform=None):
    self.hf_dataset = hf_dataset
    self.transform = transform

  def __len__(self):
    return len(self.hf_dataset)

  def __getitem__(self, idx):
    item = self.hf_dataset[idx]
    # CIFAR-10データセットでは'img'キーを使用
    image_key = "img" if "img" in item else "image"
    image = item[image_key]
    label = item["label"]

    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
      image = Image.fromarray(image)

    if self.transform:
      image = self.transform(image)

    return {"image": image, "label": label}
