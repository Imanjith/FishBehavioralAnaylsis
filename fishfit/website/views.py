from django.shortcuts import render
from moviepy.editor import VideoFileClip
from django.http import HttpResponse
from django.conf import settings
import os 
from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
# from tensorflow.keras import layers, models
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage




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



def track(request):
    video_obj = request.FILES['horses']
    location = save_video(video_obj)
    split_video(location)
    return render(request, "watch.html")
    
  
def save_video(video):
    os.mkdir("media/video")
    fs = FileSystemStorage()
    saveLocation = "video/"+ video.name 
    file = fs.save(saveLocation, video)
    path = "./media/" + file 
    return path

   
def split_video(input_video):
    os.mkdir("media/segments")
    end_time = 0
    i=0
    output_directory = "media/segments"
    clip = VideoFileClip(input_video)
    duration = clip.duration
    num_parts = int(duration // 10)
    print (duration)
    i=0
    for i in range(num_parts):
        start_time = i * 10
        end_time = (i + 1) * 10
        output_path = f"{output_directory}/seg_{i + 1}.mp4"
        segNum = "seg_" + str(i+1) + ".mp4"
        part_clip = clip.subclip(start_time, end_time)
        part_clip_without_audio = part_clip.without_audio()  # Remove audio
        part_clip_without_audio.write_videofile(output_path)
        part_clip.close()
    if end_time<duration:
        output_path = f"{output_directory}/seg_{i + 2}.mp4"
        part_clip = clip.subclip(end_time, duration)
        part_clip_without_audio = part_clip.without_audio()  # Remove audio
        part_clip_without_audio.write_videofile(output_path)
        part_clip.close()
        
    # for j in range(num_parts + 1):
    #     print(f"segments/segNum_{j+1}.mp4")
    #     results = model.track(source=f"segments/seg_{j+1}.mp4", tracker="bytetrack.yaml",show=True) 
    #     cv2.destroyAllWindows()
        
    shutil.rmtree("media/video")
    
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


# # Behavioral Analysis part
# image_height, image_width = 64, 64

# def load_data(folder_path):
#     X = []
#     y = []
#     for label, subfolder in enumerate(['normal', 'abnormal']):
#         subfolder_path = os.path.join(folder_path, subfolder)
#         for image_filename in os.listdir(subfolder_path):
#             image_path = os.path.join(subfolder_path, image_filename)
#             image = cv2.imread(image_path)
#             image = cv2.resize(image, (image_height, image_width))
#             image = image.astype('float32') / 255.0  # Normalize pixel values
#             X.append(image)
#             y.append(label)
#     return np.array(X), np.array(y)

# dataset_path = 'dataset'
# X, y = load_data(dataset_path)

# # Step 3: Split Data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Step 6: Train Model
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print('Test accuracy:', test_accuracy)

# sample_image = 'abnormal10_600_2.png'

# # Load the image
# image = cv2.imread(sample_image)

# # Check if the image is loaded successfully
# if image is None:
#     print(f"Failed to load {sample_image}")
# else:
#     # Resize the image
#     image = cv2.resize(image, (image_height, image_width))

#     # Check if the image is resized successfully
#     if image is None:
#         print(f"Failed to resize {sample_image}")
#     else:
#         # Normalize pixel values
#         image = image.astype('float32') / 255.0

#         # Add batch dimension
#         image = np.expand_dims(image, axis=0)

#         # Make prediction
#         prediction = model.predict(image)

#         # Print prediction
#         if prediction[0][0] > 0.5:
#             print(sample_image, 'is ABNORMAL')
#         else:
#             print(sample_image, 'is NORMAL')

