import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Define the paths and parameters
yolov8_model_path = "best (3).pt"
confidence_threshold = 0.3
device = "cuda:0"  # or 'cpu'

# Load the detection model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=confidence_threshold,
    device=device,
)

# Open the video capture
video_capture = cv2.VideoCapture("Golden Eagle Soaring on a Thermal.mp4")  # Use 0 for webcam or provide the video path for a specific video file

# Check if the video capture was opened successfully
if not video_capture.isOpened():
    raise IOError("Error opening video capture. Please make sure the device is connected or the video path is correct.")

# Iterate through the frames in real time
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Perform slicing-aided hyper inference on the frame
    result = get_sliced_prediction(
        frame,
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    if result and hasattr(result, 'object_prediction_list'):
        coco_predictions = result.to_coco_predictions(image_id=0)  # Convert to COCO format

        # Draw bounding boxes on the frame
        for prediction in coco_predictions:
            x, y, w, h = map(int, prediction['bbox'])  # Ensure the coordinates are integers
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{prediction['category_name']} {prediction['score']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame
    cv2.imshow('Real-time Object Detection', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()





