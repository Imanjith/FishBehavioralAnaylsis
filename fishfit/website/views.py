from django.shortcuts import render
from moviepy.editor import VideoFileClip
from django.http import HttpResponse
from django.conf import settings
import os 
from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


model = YOLO("./media/best.pt")


def home(request):
    return render(request,'index.html',{})

def about(request):
    return render(request,'about.html',{})
    
def watch(request):
    return render(request,'watch.html',{})

def blog(request):
    return render(request,'blog.html',{})

def contact(request):
    return render(request,'contact.html',{})

def contact(request):
    return render(request,'contact.html',{})

def tracke(request):
    if request.method == 'POST' and request.FILES.get('horses'):
        video_obj = request.FILES['horses']
        vid_path = video_obj.temporary_file_path()
        clip = VideoFileClip(vid_path)
        duration = clip.duration
        num_parts = int(duration // 10)
        print(duration)
        for i in range(num_parts):
            start_time = i * 10
            end_time = (i + 1) * 10
            part_clip = clip.subclip(start_time, end_time)
            tracek(part_clip)
            print(start_time,end_time)
            part_clip.close()
        if end_time<duration:
            part_clip = clip.subclip(end_time, duration)
            tracek(part_clip)
            part_clip.close()
        return render(request, 'watch.html')
    else:
        return render(request, 'watch.html')

def track(request):
    video_obj = request.FILES['horses']
    vid_path = video_obj.temporary_file_path()
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)


    # Calculate the number of frames for a 10-second clip
    num_frames_per_clip = int(fps * 10)
    frame_count = 0
    clip_index = 0
    clips = []
    # Loop through the frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # If it's the first frame of a new clip, create a new list for frames
        if frame_count % num_frames_per_clip == 0:
            clip_index += 1
            clips.append([])
        # Append the frame to the current clip's list
        clips[clip_index - 1].append(frame)
        frame_count += 1
    # Release the video capture and close any remaining video writers
    cap.release()
    cv2.destroyAllWindows()

    for clip in clips:
        tracek(clip)
    return render(request, "watch.html")
    

def tracek(cap):
    # clip_path = clip.temporary_file_path()
    # cap = cv2.VideoCapture(clip_path)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, persist=True)

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Display the annotated frame
                cv2.imshow("YOLOv8 Tracking", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    return 



  



# def track(request):
#     if request.method == 'POST':
#             video_obj = request.FILES['horses']
#             vid_path = video_obj.temporary_file_path()
#             cap = cv2.VideoCapture(vid_path)
#             while cap.isOpened():
#                 success, frame = cap.read()
#                 if success:
#                         # Run YOLOv8 tracking on the frame, persisting tracks between frames
#                         results = model.track(frame, persist=True)

#                         # Visualize the results on the frame
#                         annotated_frame = results[0].plot()

#                         # Display the annotated frame
#                         cv2.imshow("YOLOv8 Tracking", annotated_frame)

#                         # Break the loop if 'q' is pressed
#                         if cv2.waitKey(1) & 0xFF == ord("q"):
#                             break
#                 else:
#                         # Break the loop if the end of the video is reached
#                         break

#                 # Release the video capture object and close the display window
#             cap.release()
#             cv2.destroyAllWindows()
#             return 
#     else:
#         return render(request, "watch.html")




    
    







        #     cap = request.FILES["video"]
        #     # print(videoclip)
        #     results = model.track(source="./media/rail.mp4",show=True)
        #     # return render(request,"watch.html")
        #     # Load the YOLOv8 model
        #     # model = YOLO('yolov8n.pt')

        #     # Open the video file
        #     # video_path = "path/to/video.mp4"
        #     # cap = cv2.VideoCapture(video_path)

        #     # Loop through the video frames
        #     while cap.isOpened():
        #         # Read a frame from the video
        #         success, frame = cap.read()

        #         if success:
        #             # Run YOLOv8 tracking on the frame, persisting tracks between frames
        #             results = model.track(frame, persist=True)

        #             # Visualize the results on the frame
        #             annotated_frame = results[0].plot()

        #             # Display the annotated frame
        #             cv2.imshow("YOLOv8 Tracking", annotated_frame)

        #             # Break the loop if 'q' is pressed
        #             if cv2.waitKey(1) & 0xFF == ord("q"):
        #                 break
        #         else:
        #             # Break the loop if the end of the video is reached
        #             break

        #     # Release the video capture object and close the display window
        #     cap.release()
        #     cv2.destroyAllWindows()
        #     return render(request,"watch.html")

        # else:


# Behavioral Analysis part
image_height, image_width = 64, 64

def load_data(folder_path):
    X = []
    y = []
    for label, subfolder in enumerate(['normal', 'abnormal']):
        subfolder_path = os.path.join(folder_path, subfolder)
        for image_filename in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_height, image_width))
            image = image.astype('float32') / 255.0  # Normalize pixel values
            X.append(image)
            y.append(label)
    return np.array(X), np.array(y)

dataset_path = 'dataset'
X, y = load_data(dataset_path)

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 6: Train Model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', test_accuracy)

sample_image = 'abnormal10_600_2.png'

# Load the image
image = cv2.imread(sample_image)

# Check if the image is loaded successfully
if image is None:
    print(f"Failed to load {sample_image}")
else:
    # Resize the image
    image = cv2.resize(image, (image_height, image_width))

    # Check if the image is resized successfully
    if image is None:
        print(f"Failed to resize {sample_image}")
    else:
        # Normalize pixel values
        image = image.astype('float32') / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = model.predict(image)

        # Print prediction
        if prediction[0][0] > 0.5:
            print(sample_image, 'is ABNORMAL')
        else:
            print(sample_image, 'is NORMAL')

