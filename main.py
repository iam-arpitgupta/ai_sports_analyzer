import cv2 
import mediapipe as mp
import pandas as pd 
import numpy as np 
import ffmpeg
import os 
import shutil 

cap =cv2.VideoCapture('')

mp_pose = mp.soultions.pose 
pose = mp_pose.Pose()

#getting the video properties
width = int(cap.get(cv2.CAP_PROP_FROM_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FROM_HEIGHT))
fps = cap.get(cv2.CAP_FROM_FPS)

out = cv2.VideoWriter('',cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
data = []

def calculate_angles(a,b,c):
    """calcuate the angle between three points A , B , C"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2c((c[1] - b[1] ,c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - c[0]))
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle

    return angle 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare frame to rgb 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        #getting the left or the right shoulder points 
        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height,
        ]

        right_shoulder = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * width,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * height,
        ]

        left_wrist = [
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height,
        ]

        right_wrist = [
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * width,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * height,
        ]

        left_elbow= [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height,
        ]

        right_elbow = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * width,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * height,
        ]

        # cal the angle bwtween both arms 
        left_angle = calculate_angles(left_elbow,left_shoulder,left_wrist)
        right_angle = calculate_angles(right_elbow,right_shoulder,right_wrist)
        
        # store the keypoints and angles in the data list
        data.append({
            'Frame':cap.get(cv2.CAP_PROP_POS_FRAMES),
            'Left Shoulder X':left_shoulder[0], 'Left Shoulder Y':left_shoulder[1],
            'Left Elbow X':left_shoulder[0], 'Left Elbow Y':left_shoulder[1],
            'Left Wrist X':left_shoulder[0], 'Left Wrist Y':left_shoulder[1],
            'Right Shoulder X':right_shoulder[0], 'Right Shoulder Y':right_shoulder[1],
            'Right Elbow X':right_shoulder[0], 'Right Elbow Y':right_shoulder[1],
            'Right Wrist X':right_shoulder[0], 'Right Wrist Y':right_shoulder[1],
            'Left Arm Angle':left_angle,'Right Arm Angle':right_angle
        })

        # draw the landmarks and angles in data list 
        for id, lm in enumerate([left_elbow,left_shoulder,left_wrist]):
            cx, cy = int(lm[0]),int(lm[1])
            cv2.circle(frame,(cx,cy),5,(255,0,0),cv2.FILLED)
        cv2.putText(frame,str(int(left_angle)),
                    tuple(np.multiply(left_elbow,[1,1]).astype(int))
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        
        #display additional information 
        cv2.putText(frame,f'FPS: {fps}',(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
                
        #write the frame to the video 
        out.write(frame)


#creating a df using pandas 
df = pd.DataFrame(data)
file_path = '/Users/saumyagupta/Documents/game_analysis_ai/output.csv'
df.to_csv(file_path,index=False)

#extract the frames 
def extract_frames(video_path,output_folder,frame_rate=1):
    #create the output directory if it dosen't exsist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    frame_interval = int(fps / frame_rate)  # Interval between frames to save
    
    # Read and save frames
    frame_count = 0
    frame_interval = int(fps/frame_rate)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no more frames
        
        # Save frame at the specified interval
        if frame_count % frame_interval == 0:
            frame_time = frame_count / fps
            frame_filename = os.path.join(output_folder, f"frame_{int(frame_time)}")
            cv2.imwrite(frame_filename, frame)
            print(f'Saved{frame_filename}')
        
        frame_count += 1
    
    # Release video capture object
    cap.release()
    print(f"Extracted {frame_count // frame_interval} and saved frames at {output_folder} .")
    # Example usage
    extract_frames('path/to/video.mp4', 'output_frames', frame_rate=1)

shutil.make_archive('extracted_frame_nadeem','zip','extracted_frame_namdeen')

cap.release()
out.release()
cv2.destoryAllWindows()

    


