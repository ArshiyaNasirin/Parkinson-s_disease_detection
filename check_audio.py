from pathlib import Path

dataset_path = Path("dataset")

audio_files = list(dataset_path.rglob("*.wav"))

print("Total voice recordings:", len(audio_files))

print("\nFirst 10 files:")
for file in audio_files[:10]:
    print(file)