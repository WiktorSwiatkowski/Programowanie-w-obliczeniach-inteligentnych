import os
import numpy as np
from skimage import io, color, feature


def calculate_features(texture, distance, angles):
    # utworzenie macierzy zdarzeń
    glcm = feature.graycomatrix(texture, distances=[distance], angles=angles, levels=64, symmetric=True, normed=True)

    # wyznaczenie cech
    dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
    correlation = feature.graycoprops(glcm, 'correlation')[0, 0]
    contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
    energy = feature.graycoprops(glcm, 'energy')[0, 0]
    homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
    asm = feature.graycoprops(glcm, 'ASM')[0, 0]

    return [dissimilarity, correlation, contrast, energy, homogeneity, asm]


# ścieżki do katalogów z teksturami
input_paths = ["TextureSamples/Door1", "TextureSamples/Wall1", "TextureSamples/Floor1"]

# ścieżka do pliku wynikowego
output_file = "vektors.csv"

# lista cech
features_names = ["dissimilarity", "correlation", "contrast", "energy", "homogeneity", "asm"]

# lista odległości i kierunków
distances = [1, 3, 5]
angles = [0, 45, 90, 135]

with open(output_file, "w") as f:
    header = "texture_name,"
    for distance in distances:
        for angle in angles:
            header += f"{distance}_{angle}_"
            header += ",".join(features_names)
            header += ","
    header = header[:-1] + "\n"

    f.write(header)

    for i, input_path in enumerate(input_paths):

        for filename in os.listdir(input_path):

            texture = color.rgb2gray(io.imread(os.path.join(input_path, filename)))

            texture = np.uint8(texture * 63)

            features = []
            for distance in distances:
                for angle in angles:
                    angle_features = calculate_features(texture, distance, [angle, -angle])
                    features += angle_features

            f.write(  ",".join(map(str, features)) +","+f"{filename[:3]}"+ "\n")
