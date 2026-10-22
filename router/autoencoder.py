import torch
import torch.nn as nn
from dataset import WikiArtDataset
from torch.utils.data import DataLoader

class ConvAutoencoder(nn.Module):
	def __init__(self, latent_dim=256):
		super().__init__()
		
		# Enkoder: Stopniowo zmniejszamy rozdzielczość (stride=2), zwiększając liczbę kanałów
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 16, 3, stride=2, padding=1),      # B, 16, 128, 128
			nn.ReLU(),
			nn.Conv2d(16, 32, 3, stride=2, padding=1),     # B, 32, 64, 64
			nn.ReLU(),
			nn.Conv2d(32, 64, 3, stride=2, padding=1),     # B, 64, 32, 32
			nn.ReLU(),
			nn.Conv2d(64, 128, 3, stride=2, padding=1),    # B, 128, 16, 16
			nn.ReLU(),
			nn.Conv2d(128, 256, 3, stride=2, padding=1),   # B, 256, 8, 8
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(256 * 8 * 8, latent_dim)             # Kompresja do latent_dim
		)
		
		# Dekoder: Transformacja z wektora z powrotem na mapy cech
		self.decoder_linear = nn.Sequential(
			nn.Linear(latent_dim, 256 * 8 * 8),
			nn.ReLU()
		)
		
		# Dekoder: Stopniowo zwiększamy rozdzielczość (ConvTranspose2d)
		self.decoder_conv = nn.Sequential(
			nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # B, 128, 16, 16
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # B, 64, 32, 32
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # B, 32, 64, 64
			nn.ReLU(),
			nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   # B, 16, 128, 128
			nn.ReLU(),
			nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),    # B, 3, 256, 256
			nn.Sigmoid() # Zakładamy, że transform.ToTensor() normalizuje wejście do [0, 1]
		)

	def forward(self, x):
		latent = self.encoder(x)
		x_reconstructed = self.decoder_linear(latent)
		
		# Reshape do wymiarów oczekiwanych przez warstwy konwolucyjne dekodera
		x_reconstructed = x_reconstructed.view(-1, 256, 8, 8)
		x_reconstructed = self.decoder_conv(x_reconstructed)
		
		return x_reconstructed

if __name__ == "__main__":
	print("Inicjalizacja modelu...")
	model = ConvAutoencoder(latent_dim=256)
	
	print("Wczytywanie datasetu...")
	# Upewnij się, że ścieżki zgadzają się z tym, co wypakował Twój skrypt
	dataset = WikiArtDataset(csv_path="extracted_images/labels.csv", img_dir="extracted_images")
	dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
	
	print("Pobieranie batcha próbnego...")
	batch = next(iter(dataloader))
	images = batch["image"]
	
	print(f"Kształt wejścia: {images.shape}")
	outputs = model(images)
	print(f"Kształt wyjścia: {outputs.shape}")
	
	assert images.shape == outputs.shape, "Błąd: Wymiary wejścia i wyjścia się nie zgadzają!"
	print("Test zakończony sukcesem. Przepływ tensorów działa.")