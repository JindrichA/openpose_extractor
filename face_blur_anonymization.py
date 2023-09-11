import cv2
import os
import numpy as np
from tqdm import tqdm
from sys import platform

# load all npy files in folder
if platform == "win32":

    video_input_folder = r'C:\Projekty\openpose_extractor\input\PhyEx_04092023'
    npy_input_folder = r'C:\Projekty\openpose_extractor\input\PhyEx_04092023_out'
else:
    video_input_folder = r'/mnt/docker-openpose-hdd/PhyEx_04092023'
    npy_input_folder =  r'/mnt/docker-openpose-hdd/PhyEx_04092023_out'

sucessfully_processed = 0

new_folder_name = video_input_folder + "_anony"
# Check if folder doesn't exist, then create it
if not os.path.exists(new_folder_name):
    os.makedirs(new_folder_name)
    print(f"Folder {new_folder_name} created!")
else:
    print(f"Folder {new_folder_name} already exists!")


list_of_all_videos = [f for f in os.listdir(video_input_folder) if f.endswith('.mp4')]
blur_kernel_size = (31, 31)
rectangle_size = (80, 80)
width, height = rectangle_size
for filename in tqdm(list_of_all_videos):

    i = 0

    check_if_final_product_exists = new_folder_name + '/' + filename + '_comp.mp4'
    if os.path.exists(check_if_final_product_exists):
        print(f"{check_if_final_product_exists} exists. Skipping this iteration.")
        continue



    # Load a video
    # video = cv2.VideoCapture(video_input_folder + '\\' + filename[:-4]+'.mp4')
    video = cv2.VideoCapture(os.path.join(video_input_folder, filename[:-4]+'.mp4'))


    # get width and height
    width_original = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_original = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    last_frame = np.ones((height_original, width_original, 3), np.uint8) * 255

    # get fps of video
    fps = video.get(cv2.CAP_PROP_FPS)

    try:
        filename_npy = filename[:-4] + '.npy'
        # pose_data = np.load(npy_input_folder + '\\' + filename_npy)
        pose_data = np.load(os.path.join(npy_input_folder, filename_npy))
    except:
        print(f"Failed to import {filename_npy}. Continuing to next file.")
        continue


    # create a new video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(new_folder_name+'/'+filename[:-4]+'_anony.avi',fourcc, fps, (width_original, height_original))


    # Video frame count
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # read pose data from npy file




    # Read a frame from the video
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Convert the frame to RGB format
        # iterate i
        i += 1

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Run the pose estimation on the frame
        # plot a circle to the frame to location given by the lanmark saved in the npy file
        try:
            x = int(pose_data[i,0,0])
            y = int(pose_data[i,0,1])
            top_left = (x - width // 2, y - height // 2)
            bottom_right = (x + width // 2, y + height // 2)
            roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            blurred_roi = cv2.GaussianBlur(roi, blur_kernel_size, 0)
            frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blurred_roi
          #  cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        except:

            #
            try:
                bottom_right = bottom_right_last
                roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                blurred_roi = cv2.GaussianBlur(roi, blur_kernel_size, 0)
                frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blurred_roi
            except:
                #frame = np.full((height_original, width_original, 3), 255, dtype=np.uint8)
                frame = last_frame
        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(frame)
        # create a copy of frame
        bottom_right_last = bottom_right
        last_frame = frame.copy()
    # Release the video and close the CSV file
    video.release()
    cv2.destroyAllWindows()
    out.release()



    if platform == "win32":
        puvodni_soubor = new_folder_name+'/'+filename[:-4]+'_anony.avi'
        command_ffmpeg = "ffmpeg -i " + puvodni_soubor + " -vcodec h264 -acodec aac " + new_folder_name + '/' + filename + '_comp.mp4'
    else:
        puvodni_soubor = new_folder_name+'/'+filename[:-4]+'_anony.avi'
        command_ffmpeg = "ffmpeg -i " + puvodni_soubor + " -vcodec h264 -acodec aac " + new_folder_name + '/' + filename + '_comp.mp4'
    try:
        os.system(command_ffmpeg)
        os.remove(puvodni_soubor)
        sucessfully_processed =sucessfully_processed +1
    except:
        continue
print(f'Job done - anonymization (blur face) for {sucessfully_processed} files. ')




