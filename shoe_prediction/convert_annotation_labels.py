import os

# Paths
image_dir = "F:/malathi/AI/DEEP_LEARNING/Shoe_detect_2/test/images"
label_dir = os.path.join(image_dir, "labels")

# Ensure label directory exists
os.makedirs(label_dir, exist_ok=True)

# Function to convert to YOLO format
def convert_to_yolo(img_width, img_height, x_min, y_min, x_max, y_max):
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return f"{x_center} {y_center} {width} {height}"

# Read annotation file
annotation_file = "F:\\malathi\\AI\\DEEP_LEARNING\\Shoe_detect_2\\test\\images\\_annotations.txt"

with open(annotation_file, "r") as file:
    for line in file:
        parts = line.strip().split()
        image_name = parts[0]  # Image name
        annotations = parts[1:]  # Rest are bounding boxes

        # Get image dimensions
        img_path = os.path.join(image_dir, image_name)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        from PIL import Image
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Convert all boxes
        yolo_annotations = []
        for ann in annotations:
            x_min, y_min, x_max, y_max, class_id = map(int, ann.split(","))
            yolo_ann = f"{class_id} {convert_to_yolo(img_width, img_height, x_min, y_min, x_max, y_max)}"
            yolo_annotations.append(yolo_ann)

        # Save YOLO format labels
        label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as label_file:
            label_file.write("\n".join(yolo_annotations))

print("✅ Annotations converted successfully!")
