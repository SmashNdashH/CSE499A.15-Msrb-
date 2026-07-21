from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolo11n.pt')  # You can choose yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.

# Train the model 
results = model.train(data='code/xview-buildings.yaml', epochs=50, imgsz=640, save=True)
