from ultralytics import YOLO
import os
import numpy as np
from upload_to_s3 import upload_file_to_s3
import tempfile

def run(image_path, output_dir):
    model = YOLO(os.path.join(os.path.dirname(__file__), "yolo", "best_200.pt"))

    results = model.predict(source=image_path)

    filename = os.path.basename(image_path)
    output_filename = f"{output_dir}/yolo_{filename}"

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        results[0].save(tmp.name)
        image_url = upload_file_to_s3(tmp.name, output_filename)

    fire_probs = []
    for result in results:
        for box in result.boxes:
            if box.cls == 0 or box.cls == 1:
                fire_probs.append(box.conf.item())

    return {
        "path": image_url,
        "max_prob": max(fire_probs) if fire_probs else 0,
        "mean_prob": np.mean(fire_probs) if fire_probs else 0,
        "fire_count": len(fire_probs),
    }
