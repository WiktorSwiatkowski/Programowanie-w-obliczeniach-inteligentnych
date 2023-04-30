import os

from PIL import Image

# ścieżka do katalogów z obrazkami
input_paths = [
    "TextureSamples/Door",
    "TextureSamples/Floor",
    "TextureSamples/Wall"
]
# ścieżka do nowych katalogów
output_paths = [
    "TextureSamples/Door1",
    "TextureSamples/Floor1",
    "TextureSamples/Wall1"
]

# utworzenie nowych katalogów, jeśli nie istnieją
for output_path in output_paths:
    if not os.path.exists(output_path):
        os.makedirs(output_path)

# pętle po folderach z obrazkami i odpowiadających nowych katalogach
for input_path, output_path in zip(input_paths, output_paths):
    # pętla po plikach w folderze z obrazkami
    for filename in os.listdir(input_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # otwarcie obrazka
            img = Image.open(os.path.join(input_path, filename))

            # wycięcie fragmentów 128x128
            width, height = img.size
            for i in range(0, width, 128):
                for j in range(0, height, 128):
                    box = (i, j, i+128, j+128)
                    region = img.crop(box)

                    # zapisanie fragmentu do pliku
                    region.save(os.path.join(output_path, f"{filename}_{i}_{j}.png"))
