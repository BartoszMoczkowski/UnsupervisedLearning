import os
import torch
import polars as pl
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class WikiArtDataset(Dataset):
	def __init__(self, csv_path, img_dir, transform=None):
		# Szybkie wczytanie całego pliku CSV do pamięci
		df = pl.read_csv(csv_path)
		
		# Konwersja całych kolumn na natywne listy dla maksymalnej szybkości w __getitem__
		self.filenames = df["filename"].to_list()
		self.styles = df["style"].to_list()
		self.artists = df["artist"].to_list()
		
		self.img_dir = img_dir
		
		if transform:
			self.transform = transform
		else:
			self.transform = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
			])

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		# Szybkie indeksowanie zwykłej listy pythona bez narzutu Pandas/Polars
		img_name = self.filenames[idx]
		img_path = os.path.join(self.img_dir, img_name)
		
		with Image.open(img_path) as img:
			img_rgb = img.convert("RGB")
			tensor_img = self.transform(img_rgb)

		return {
			"image": tensor_img,
			"style": self.styles[idx],
			"artist": self.artists[idx]
		}