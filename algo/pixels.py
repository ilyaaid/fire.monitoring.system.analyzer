import numpy as np
from PIL import Image
import os


def rgb_to_hsv(rgb_image):
    norm_image = rgb_image / 255.0
    R, G, B = norm_image[:, :, 0], norm_image[:, :, 1], norm_image[:, :, 2]

    Cmax = np.maximum(np.maximum(R, G), B)
    Cmin = np.minimum(np.minimum(R, G), B)
    delta = Cmax - Cmin

    H = np.zeros_like(Cmax)

    mask_R = (delta != 0) & (Cmax == R)
    mask_G = (delta != 0) & (Cmax == G)
    mask_B = (delta != 0) & (Cmax == B)

    H[mask_R] = ((G[mask_R] - B[mask_R]) / delta[mask_R]) % 6
    H[mask_G] = (B[mask_G] - R[mask_G]) / delta[mask_G] + 2
    H[mask_B] = (R[mask_B] - G[mask_B]) / delta[mask_B] + 4

    H = H * 60

    S = np.zeros_like(Cmax)
    np.divide(delta, Cmax, out=S, where=Cmax != 0)

    V = Cmax

    hsv_image = np.stack((H, S, V), axis=-1)
    return hsv_image


def detect_fire_by_color(img_array):
    hsv_array = rgb_to_hsv(img_array)

    H = hsv_array[:, :, 0]
    S = hsv_array[:, :, 1]
    V = hsv_array[:, :, 2]

    fire_mask = ((H < 30) | (H > 350)) & (S > 0.6) & (V > 0.5)

    fire_pixels = np.zeros_like(img_array)
    fire_pixels[fire_mask] = img_array[fire_mask]

    fire_area = np.sum(fire_mask)
    total_pixels = fire_mask.size
    fire_percentage = (fire_area / total_pixels) * 100

    if fire_area > 0:
        print(
            f"Обнаружен огонь! Площадь: {fire_area} пикселей ({fire_percentage:.2f}%)"
        )
    else:
        print("Огонь не обнаружен.")

    return fire_pixels, fire_mask


def binary_erosion(image_array, kernel_radius=1):
    height, width = image_array.shape
    eroded = np.zeros_like(image_array)

    for y in range(kernel_radius, height - kernel_radius):
        for x in range(kernel_radius, width - kernel_radius):
            neighborhood = image_array[
                y - kernel_radius : y + kernel_radius + 1,
                x - kernel_radius : x + kernel_radius + 1,
            ]
            eroded[y, x] = 1 if np.all(neighborhood == 1) else 0

    return eroded


def binary_dilation(image_array, kernel_radius=1):
    height, width = image_array.shape
    dilated = np.zeros_like(image_array)

    for y in range(kernel_radius, height - kernel_radius):
        for x in range(kernel_radius, width - kernel_radius):
            neighborhood = image_array[
                y - kernel_radius : y + kernel_radius + 1,
                x - kernel_radius : x + kernel_radius + 1,
            ]
            dilated[y, x] = 1 if np.any(neighborhood == 1) else 0

    return dilated


def binary_opening(image_array, kernel_radius=1):
    eroded = binary_erosion(image_array, kernel_radius)
    opened = binary_dilation(eroded, kernel_radius)
    return opened


def binary_closing(image_array, kernel_radius=1):
    dilated = binary_dilation(image_array, kernel_radius)
    closed = binary_erosion(dilated, kernel_radius)
    return closed


def binary_opening_closing(image_array, kernel_radius=1):
    opened = binary_opening(image_array, kernel_radius)
    closed = binary_closing(opened, kernel_radius)
    return closed


def calculate_white_percentage(binary_array):
    total_pixels = binary_array.size
    white_pixels = np.sum(binary_array)
    return (white_pixels / total_pixels) * 100


def apply_morphology_preserve_color(original_img, binary_array):
    result_img = original_img.copy()
    result_img_array = np.array(result_img)
    result_img_array[binary_array == 0] = 0
    return Image.fromarray(result_img_array)


def run(image_path, output_dir, origin):
    os.makedirs(output_dir, exist_ok=True)

    timestamp_folder = os.path.basename(output_dir)
    base_results_folder = os.path.basename(os.path.dirname(output_dir))

    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img_array = np.array(img)

    fire_pixels, fire_mask = detect_fire_by_color(img_array)
    fire_img = Image.fromarray(fire_pixels)

    base_name = os.path.basename(image_path)

    img_gray = fire_img.convert("L")
    img_binary = np.array(img_gray) > 0

    operations = {
        "eroded": binary_erosion(img_binary, kernel_radius=2),
        "dilated": binary_dilation(img_binary, kernel_radius=2),
        "opened": binary_opening(img_binary, kernel_radius=2),
        "closed": binary_closing(img_binary, kernel_radius=2),
        "opened_closed": binary_opening_closing(img_binary, kernel_radius=2),
    }

    results = {}

    for name, result in operations.items():
        processed_img = apply_morphology_preserve_color(img, result)
        filename = f"{name}_{base_name}"
        processed_path = os.path.join(output_dir, filename)
        processed_img.save(processed_path)
        print(f"Результат {name} сохранен в: {processed_path}")

        percent = calculate_white_percentage(result)
        print(f"{name}: {percent:.2f}% белых пикселей")

        relative_url = f"{timestamp_folder}/{filename}"
        processed_url = f"{origin}results/{relative_url}"

        results[name] = {"path": processed_url, "white_percentage": percent}

    return results
