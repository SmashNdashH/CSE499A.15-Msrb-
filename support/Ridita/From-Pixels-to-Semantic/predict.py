from ultralytics import YOLO

# Load a pretrained model (or your trained model e.g., 'runs/detect/train/weights/best.pt')
model = YOLO('yolo11n.pt')

# Run prediction on the images in the data/images folder
model.predict('data/images', imgsz=640, save=True)
