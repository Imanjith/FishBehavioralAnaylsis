from django.shortcuts import render
from moviepy.editor import VideoFileClip
from django.http import HttpResponse
from django.conf import settings
import os 
from ultralytics import YOLO
import cv2
import numpy as np



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