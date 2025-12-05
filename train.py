from ultralytics import YOLO

def train_model():
    # Load a model
    model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # data argument points to the dataset directory containing train/val/test folders
    results = model.train(data='/Users/rajasivaranjan/cotton_disease_model/yolo_dataset', epochs=10, imgsz=224)
    
    # Validate the model
    metrics = model.val()
    print(f"Validation metrics: {metrics}")
    
    # Export the model
    success = model.export(format='onnx')
    print(f"Model exported: {success}")

if __name__ == '__main__':
    train_model()
