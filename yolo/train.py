from ultralytics import YOLO

# Load a model
# model = YOLO("yolo10l.pt")
# model = YOLO("./yolov10l.yaml")

# Train the model

#fold train
for i in range(1, 6):
    model = YOLO("./yolov8l.pt")
    train_results = model.train(
    data=f"./recycle-trash{i}.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=1024,  # training image size
    device='cuda',  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch=16,
    patience=10,
    name=f"yolov8-fold[{i}]_2"
)

# Evaluate model performance on the validation set
# metrics = model.val()
