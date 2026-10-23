#!/usr/bin/env python3
import os
import glob
import polars as pl
from pathlib import Path

# Ścieżki
PARQUET_DIR = "../wikiart-parquet/default/partial-train"
OUT_IMG_DIR = "./extracted_images"

def main():
	parquet_files = sorted(glob.glob(f"{PARQUET_DIR}/*.parquet"))
	if not parquet_files:
		print(f"Błąd: Nie znaleziono plików .parquet w ścieżce {PARQUET_DIR}")
		return

	os.makedirs(OUT_IMG_DIR, exist_ok=True)
	
	metadata_records = []
	global_idx = 0

	for file_path in parquet_files:
		print(f"Przetwarzanie pliku: {os.path.basename(file_path)}...")
		
		# Wczytanie całego pliku do pamięci
		df = pl.read_parquet(file_path)
		
		# Iteracja po wierszach i zapis na dysk
		for row in df.iter_rows(named=True):
			filename = f"{global_idx:07d}.jpg"
			img_path = Path(OUT_IMG_DIR) / filename
			
			# Zapis bajtów na dysk jako .jpg
			if not img_path.exists():
				img_path.write_bytes(row["image"]["bytes"])
			
			# Zbieranie metadanych do etykiet
			metadata_records.append({
				"id": global_idx,
				"filename": filename,
				"style": row.get("style", ""),
				"artist": row.get("artist", ""),
				"genre": row.get("genre", "")
			})
			
			global_idx += 1
			
			if global_idx % 1000 == 0:
				print(f"  Wypakowano {global_idx} obrazów...")

	# Zapisanie połączonych etykiet do jednego pliku CSV
	print("Trwa zapisywanie pliku labels.csv...")
	meta_df = pl.DataFrame(metadata_records)
	meta_csv_path = Path(OUT_IMG_DIR) / "labels.csv"
	meta_df.write_csv(meta_csv_path)

	print(f"\nGotowe! Wypakowano łącznie {global_idx} obrazów do {OUT_IMG_DIR}")
	print(f"Plik z etykietami zapisano w: {meta_csv_path}")

if __name__ == "__main__":
	main()