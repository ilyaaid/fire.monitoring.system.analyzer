from ultralytics import YOLO
import os
import numpy as np

def run(image_path, output_dir, origin):
    os.makedirs(output_dir, exist_ok=True)

    timestamp_folder = os.path.basename(output_dir)
    base_results_folder = os.path.basename(os.path.dirname(output_dir))

    filename = os.path.basename(image_path)
    dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(dir, "yolo", "best.pt")
    
    model = YOLO(model_path)
    results = model.predict(source=image_path)
    
    output_filename = 'yolo_'+filename
    # Сохраняем результат
    final_path = os.path.join(output_dir, output_filename)
    results[0].save(final_path)
    
    # Собираем статистику по обнаруженным пожарам
    fire_probs = []
    for result in results:
        for box in result.boxes:
            if box.cls == 0:  # предполагаем, что класс 0 - это "огонь"
                fire_probs.append(box.conf.item())

    relative_url = f"{timestamp_folder}/{output_filename}"
    image_url = f"{origin}results/{relative_url}"
    
    # Рассчитываем статистику
    result_stats = {
        "path": image_url,
        "max_prob": max(fire_probs) if fire_probs else 0,
        "mean_prob": np.mean(fire_probs) if fire_probs else 0,
        "fire_count": len(fire_probs)
    }
    
    return result_stats
