from flask import Flask, jsonify, send_from_directory, request
import algo.pixels as px
import algo.yolo as yolo
import os
import random
from datetime import datetime
import shutil
import argparse
from urllib.parse import urlparse

app = Flask(__name__)

FIRE_IMAGES_DIR = "images/fire"
NOFIRE_IMAGES_DIR = "images/nofire"

MOSCOW_LATITUDE = 55.638062
MOSCOW_LONGITUDE = 31.882613
LATITUDE = MOSCOW_LATITUDE + random.uniform(-0.3, 0.3)
LONGITUDE = MOSCOW_LONGITUDE + random.uniform(-0.3, 0.3)


def get_random_image():
    if random.random() > 0.5:
        image_dir = FIRE_IMAGES_DIR
        image_type = "fire"
    else:
        image_dir = NOFIRE_IMAGES_DIR
        image_type = "nofire"

    images = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not images:
        return None, None, None

    image_name = random.choice(images)
    image_path = os.path.join(image_dir, image_name)

    return image_path, image_name, image_type


def clean_old_results(results_dir, max_age_minutes=5):
    now = datetime.now()
    for folder_name in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder_name)
        if os.path.isdir(folder_path):
            try:
                folder_time = datetime.strptime(folder_name, "%Y%m%d_%H%M%S")
                age_minutes = (now - folder_time).total_seconds() / 60
                if age_minutes > max_age_minutes:
                    shutil.rmtree(folder_path)
            except ValueError:
                continue


@app.route("/results/<path:filename>")
def results(filename):
    print(app.config["RESULTS_DIR"])
    return send_from_directory(app.config["RESULTS_DIR"], filename)


@app.route("/analyze", methods=["GET"])
def analyze_image():
    base_results_dir = app.config["RESULTS_DIR"]
    clean_old_results(base_results_dir, max_age_minutes=300)

    origin = request.url_root
    image_path, image_name, image_type = get_random_image()

    if not image_path:
        return jsonify({"error": "No images found"}), 404

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_subdir = os.path.join(base_results_dir, timestamp)
    os.makedirs(result_subdir, exist_ok=True)

    print(f"[INFO] Saving results to: {result_subdir}")

    original_in_results = os.path.join(result_subdir, image_name)
    shutil.copy2(image_path, original_in_results)

    px_results = px.run(image_path, result_subdir, origin)
    yolo_results = yolo.run(image_path, result_subdir, origin)

    original_image_url = f"{origin}results/{timestamp}/{image_name}"

    response = {
        "id": str(urlparse(request.host_url).port),
        "coordinates": {"latitude": LATITUDE, "longitude": LONGITUDE},
        "timestamp": datetime.now().isoformat(),
        "image_info": {
            "path": original_image_url,
            "name": image_name,
            "type": image_type,
        },
        "results": {"pixels": px_results, "yolo": yolo_results},
        "fire": bool(
            px_results["opened_closed"]["white_percentage"] > 0
            or yolo_results["fire_count"] > 0
        ),
    }

    return jsonify(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask server for image analysis")
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)",
    )
    args = parser.parse_args()

    # Создание папки results_<порт>
    RESULTS_DIR = f"results_{args.port}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    app.config["RESULTS_DIR"] = RESULTS_DIR

    print(f"[START] Server running on port {args.port}")
    print(f"[CONFIG] Results will be saved to: {RESULTS_DIR}")

    app.run(host="0.0.0.0", port=args.port, debug=True)
