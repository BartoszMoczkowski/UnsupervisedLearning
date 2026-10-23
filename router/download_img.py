#!/usr/bin/env python3
import os
import glob
import polars as pl
from pathlib import Path

# Ścieżki - dostosuj jeśli wypakowane pliki parquet masz w innym miejscu
PARQUET_DIR = "../wikiart-parquet/default/partial-train"
OUT_IMG_DIR = "./prototype_images"
NUM_IMAGES = 1000

def main():
	# Szukamy pierwszego pliku parquet z danymi
	parquet_files = sorted(glob.glob(f"{PARQUET_DIR}/*.parquet"))
	if not parquet_files:
		print(f"Błąd: Nie znaleziono plików .parquet w ścieżce {PARQUET_DIR}")
		return

	first_parquet = parquet_files[0]
	print(f"Pobieranie próbek metadanych i obrazów z: {first_parquet}")

	# Tworzymy nowy folder na czysty prototyp
	os.makedirs(OUT_IMG_DIR, exist_ok=True)

	# Leniwe wczytanie tylko określonej liczby wierszy (szybkie i nie zapycha RAM-u)
	lf = pl.scan_parquet(first_parquet).slice(0, NUM_IMAGES)
	df = lf.collect()

	metadata_records = []

	# Ekstrakcja obrazów i etykiet w jednym przebiegu
	for idx, row in enumerate(df.iter_rows(named=True)):
		filename = f"{idx:07d}.jpg"
		img_path = Path(OUT_IMG_DIR) / filename
		
		# Zapis bajtów na dysk jako .jpg
		if not img_path.exists():
			img_path.write_bytes(row["image"]["bytes"])
		
		# Zapis powiązanych etykiet (ignorujemy surowe bajty z dict'a)
		metadata_records.append({
			"id": idx,
			"filename": filename,
			"style": row["style"],
			"artist": row["artist"],
			"genre": row["genre"]
		})

	# Konwersja listy słowników do ramki danych Polars i zapis
	meta_df = pl.DataFrame(metadata_records)
	meta_csv_path = Path(OUT_IMG_DIR) / "labels.csv"
	meta_df.write_csv(meta_csv_path)

	print(f"Gotowe! Zapisano {len(meta_df)} obrazów oraz etykiety w: {meta_csv_path}")

if __name__ == "__main__":
	main()