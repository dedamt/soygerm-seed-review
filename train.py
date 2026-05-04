from ultralytics import YOLO

def main():
    model = YOLO("yolo11n-seg.pt")

    model.train(
        task="segment",
        data="data.yaml",
        epochs=50,
        batch=2,
        imgsz=1024,
        device="cpu",
        workers=0,
        name="teste_yolo11n_seg_soja",
        pretrained=True,
        plots=True
    )

if __name__ == "__main__":
    main()