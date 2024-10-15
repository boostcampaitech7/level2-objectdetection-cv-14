from ultralytics import YOLO

# Load a model
model = YOLO("yolo11l.pt")

# Train the model
train_results = model.train(
    data="./recycle-trash.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=1024,  # training image size
    device='cuda',  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    # multi_scale=True,
    patience=7,
    batch=8,
    iou=0.6,
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
# results = model("path/to/image.jpg")
# results[0].show()

# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model