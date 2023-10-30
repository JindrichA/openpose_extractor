import cv2
import os
import numpy as np
from tqdm import tqdm
from sys import platform

def load_pose_data(file_path):
    try:
        return np.load(file_path)
    except FileNotFoundError:
        print(f"Failed to import {file_path}. Continuing to next file.")
        return None

def main():
    if platform == "win32":
        video_input_folder = r'C:\Projekty\openpose_extractor\input\PhyEx_04092023'
        npy_input_folder = r'C:\Projekty\openpose_extractor\input\PhyEx_04092023_out'
    else:
        video_input_folder = '/mnt/docker-openpose-hdd/Data/PhyEx_EXp12102023'
        npy_input_folder =  '/mnt/docker-openpose-hdd/Data/PhyEx_EXp12102023_out'

    successfully_processed = 0
    new_folder_name = video_input_folder + "_anony"
    os.makedirs(new_folder_name, exist_ok=True)
    print(f"Processing to folder {new_folder_name}")

    list_of_all_videos = [f for f in os.listdir(video_input_folder) if f.endswith('.mp4')]
    blur_kernel_size = (31, 31)
    rectangle_size = (80, 80)
    width, height = rectangle_size

    for filename in tqdm(list_of_all_videos):
        final_product_path = os.path.join(new_folder_name, filename + '_comp.mp4')
        if os.path.exists(final_product_path):
            print(f"{final_product_path} exists. Skipping this iteration.")
            continue

        video_path = os.path.join(video_input_folder, filename)
        video = cv2.VideoCapture(video_path)
        width_original = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_original = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        pose_data = load_pose_data(os.path.join(npy_input_folder, filename[:-4] + '.npy'))
        if pose_data is None:
            continue

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = os.path.join(new_folder_name, filename[:-4] + '_anony.avi')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width_original, height_original))
        last_frame = np.ones((height_original, width_original, 3), np.uint8) * 255
        bottom_right_last = None

        for i, _ in enumerate(pose_data):
            ret, frame = video.read()
            if not ret:
                break

            try:
                x, y = int(pose_data[i, 0, 0]), int(pose_data[i, 0, 1])
                top_left = (x - width // 2, y - height // 2)
                bottom_right = (x + width // 2, y + height // 2)
                roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                blurred_roi = cv2.GaussianBlur(roi, blur_kernel_size, 0)
                frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blurred_roi
            except:
                if bottom_right_last:
                    roi = frame[top_left[1]:bottom_right_last[1], top_left[0]:bottom_right_last[0]]
                    blurred_roi = cv2.GaussianBlur(roi, blur_kernel_size, 0)
                    frame[top_left[1]:bottom_right_last[1], top_left[0]:bottom_right_last[0]] = blurred_roi
                else:
                    frame = last_frame

            out.write(frame)
            bottom_right_last = bottom_right
            last_frame = frame.copy()

        video.release()
        out.release()

        output_file = os.path.join(new_folder_name, filename[:-4] + '_comp.mp4')
        command_ffmpeg = ["ffmpeg", "-i", output_path, "-vcodec", "h264", "-acodec", "aac", output_file]
        if os.system(' '.join(command_ffmpeg)) == 0:
            os.remove(output_path)
            successfully_processed += 1
        else:
            print(f"Some problem with {output_path}")

    print(f'Job done - anonymization (blur face) for {successfully_processed} files.')

if __name__ == '__main__':
    main()
