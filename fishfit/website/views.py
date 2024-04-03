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
from collections import defaultdict
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models



model = YOLO("yolov8n.pt")


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

def fishs(request):
    return render(request,'fishs.html',{})

def fishs(request):
    return render(request,'fishs.html',{})
    
def testimonial(request):
    return render(request,'testimonial.html',{})



def track(request):
    # video_obj = request.FILES['horses']
    # location = save_video(video_obj)
    # enhance_video(location)
    # clip = VideoFileClip(location)
    # duration = clip.duration
    # num_parts=0
    # if duration>10:
    #     num_parts = split_video(location)
    # track_video(num_parts)
    results = analyse()
    return render(request, "results.html", {'price': results})
    
    

def singleScaleRetinex(img, sigma):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex

def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex

def colorRestoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration

def simplestColorBalance(img, low_clip, high_clip):
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
    return img

def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, sigma_list)
    img_color = colorRestoration(img, alpha, beta)
    img_msrcr = G * (img_retinex * img_color + b)
    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255
    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)
    return img_msrcr

def automatedMSRCR(img):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, [15, 80, 250])
    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)
    img_retinex = cv2.fastNlMeansDenoisingColored(img_retinex,None,20,20,7,21)
    return img_retinex

def MSRCP(img, sigma_list, low_clip, high_clip):
    img = np.float64(img) + 1.0
    intensity = np.sum(img, axis=2) / img.shape[2]
    retinex = multiScaleRetinex(intensity, sigma_list)
    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)
    intensity1 = simplestColorBalance(retinex, low_clip, high_clip)
    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0
    img_msrcp = np.zeros_like(img)
    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]
    img_msrcp = np.uint8(img_msrcp - 1.0)
    return img_msrcp

def enhance_video(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object to save enhanced video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter('enhanced_video.mp4', fourcc, fps, (width, height))

    # Process each frame of the video
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Apply the automatedMSRCR enhancement method
        enhanced_frame = automatedMSRCR(frame)

        # Write the enhanced frame to the output video
        output.write(enhanced_frame)

    # Release video capture and writer objects
    video.release()
    output.release()

  
def save_video(video):
    if os.path.isdir('media/video'):
        shutil.rmtree('media/video')
    os.mkdir("media/video")
    fs = FileSystemStorage()
    saveLocation = "video/"+ video.name 
    file = fs.save(saveLocation, video)
    path = "./media/" + file 
    return path


   
def split_video(input_video):
    output_directory = "media/segments"

    if os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.mkdir(output_directory)
    end_time = 0
    i=0
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
        
    shutil.rmtree("media/video")
    return num_parts



def track_video(num):
    os.mkdir("media/movements")

    for j in range(num + 1):
        track_history = defaultdict(lambda: [])
        complete_tracks = defaultdict(list)
        cap = cv2.VideoCapture(f"media/segments/seg_{j+1}.mp4")

    
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, persist=True)

                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                # track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.int().cpu().tolist()


                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)
                    
                for track_id, track in track_history.items():
                    complete_tracks[track_id].extend(track)

                # Display the annotated frame
                cv2.imshow("YOLOv8 Tracking", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        cap.release()
        cv2.destroyAllWindows()
        
        
        plt.figure(figsize=(10, 6), facecolor="black")
        plt.axis('off')
        trajectories = list(track_history.values())

        for trajectory in trajectories:
            x_values, y_values = zip(*trajectory)
            plt.plot(x_values, y_values)

        plt.title("Object Trajectories")
        plt.savefig(f"media/movements/seg_{j+1}.png")
        
        return num


        

    shutil.rmtree("media/segments")
    
    
def analyse(num=4):
    normal_count = 0
    abnormal_count = 0
    
    for j in range(num + 1):
        image_height, image_width = 64, 64
        model = models.load_model('media/analysis.keras')

        sample_image = f"media/movements/seg_{j+1}.png"

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
                    abnormal_count+=1
                else:
                    print(sample_image, 'is NORMAL')
                    normal_count+=1
                    
                    
    results = [abnormal_count,normal_count]
    print(results)
    return results


    
    



