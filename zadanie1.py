import os
import urllib.request
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Wczytanie obrazu rastrowego (lokalnego lub zdalnego z użyciem urllib)
local_path = "obraz.jpg"
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Oranges_and_lemon.jpg/640px-Oranges_and_lemon.jpg"

if not os.path.exists(local_path):
    print(f"Pobieranie obrazu zdalnego z: {image_url}")
    req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(local_path, "wb") as out_file:
        out_file.write(response.read())

# Wczytanie obrazu za pomocą OpenCV (domyślnie w formacie BGR)
image_bgr = cv2.imread(local_path)
# Konwersja BGR -> RGB do poprawnego wyświetlenia w matplotlib
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

print(f"Oryginalne wymiary obrazu (wysokość, szerokość, kanały): {image_rgb.shape}")

# 2. Wyświetlenie oryginalnego obrazu z użyciem matplotlib
plt.figure(figsize=(6, 5))
plt.imshow(image_rgb)
plt.title(f"Obraz oryginalny ({image_rgb.shape[1]}x{image_rgb.shape[0]} px)")
plt.axis("off")
plt.tight_layout()
plt.savefig("1_obraz_oryginalny.png")
plt.show()

# 3. Zmiana rozdzielczości (zmniejszenie o 50%) oraz konwersja na skalę szarości (grayscale)
new_width = int(image_bgr.shape[1] * 0.5)
new_height = int(image_bgr.shape[0] * 0.5)
resized_image = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)

gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
print(f"Wymiary po zmniejszeniu o 50% i konwersji na grayscale: {gray_image.shape}")

# 4. Obrót obrazu o 90 stopni (zgodnie z ruchem wskazówek zegara)
rotated_image = cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE)
print(f"Wymiary po obrocie o 90 stopni: {rotated_image.shape}")

# 5. Wyświetlenie obrazu wynikowego
plt.figure(figsize=(5, 6))
plt.imshow(rotated_image, cmap="gray")
plt.title(f"Obraz wynikowy: 50%, grayscale, obrót 90° ({rotated_image.shape[1]}x{rotated_image.shape[0]} px)")
plt.axis("off")
plt.tight_layout()
plt.savefig("2_obraz_wynikowy.png")
plt.show()

# 6. Wyświetlenie macierzy obrazu w postaci tablicy liczb
print("\nMacierz obrazu wynikowego (tablica liczb):")
print(rotated_image)
