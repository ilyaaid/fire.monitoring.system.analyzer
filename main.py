from flask import Flask, jsonify, request
import algo.pixels as px
import algo.yolo as yolo
import os
import random
from datetime import datetime
import argparse
from upload_to_s3 import upload_file_to_s3

app = Flask(__name__)

FIRE_IMAGES_DIR = "images/fire"
NOFIRE_IMAGES_DIR = "images/nofire"


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


@app.route("/analyze", methods=["GET"])
def analyze_image():
    [host, port] = request.host.split(':')
    image_path, image_name, image_type = get_random_image()

    if not image_path:
        return jsonify({"error": "No images found"}), 404

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_subdir = f"{host}_{port}/{timestamp}"
    
    original_s3_key = f"{s3_subdir}/{image_name}"
    original_image_url = upload_file_to_s3(image_path, original_s3_key)

    if not original_image_url:
        return jsonify({"error": "Не удалось загрузить оригинальное изображение"}), 500

    px_results = px.run(image_path, s3_subdir)
    yolo_results = yolo.run(image_path, s3_subdir)

    response = {
        "timestamp": datetime.now().isoformat(),
        "image_info": {
            "path": original_image_url,
            "name": image_name,
            "type": image_type,
        },
        "results": {
            "pixels": px_results,
            "yolo": yolo_results,
        },
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

    RESULTS_DIR = f"results_{args.port}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    app.config["RESULTS_DIR"] = RESULTS_DIR

    print(f"[START] Server running on port {args.port}")
    print(f"[CONFIG] Results will be saved to: {RESULTS_DIR}")

    app.run(host="0.0.0.0", port=args.port, debug=True)
