import os
import sys
import argparse
import cv2
import numpy as np
import scipy.io
from tqdm import tqdm
import glob
import shutil
import importlib
import pandas as pd

def get_arguments():
    parser = argparse.ArgumentParser(description="Process input arguments.")
    parser.add_argument("--input_folder", type=str, default="test", help="Input folder")
    return parser.parse_args()






keypoint_names = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
    "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe",
    "RSmallToe", "RHeel"
]
def create_dataframe(k, keypoint_names):
    data = []
    for i, name in enumerate(keypoint_names):
        df = pd.DataFrame({
            f'{name}_x': k[:, i, 0],
            f'{name}_y': k[:, i, 1],
            f'{name}_c': k[:, i, 2]
        })
        data.append(df)
    return pd.concat(data, axis=1)


def setup_openpose(platform, dir_path):
    global op  # Declare the module object at the global scope

    if platform == "win32":
        sys.path.append(os.path.join(dir_path, 'python', 'openpose', 'Release'))
        os.environ['PATH'] += ';' + os.path.join(dir_path, 'x64', 'Release') + ';' + dir_path + '/bin;'
        op_module = importlib.import_module('pyopenpose')
        op = op_module
    else:
        sys.path.extend([
            '/mnt/docker-openpose/openpose/build/python',
            '/mnt/docker-openpose/openpose'
        ])
        # Import using importlib
        op_module = importlib.import_module('openpose.pyopenpose')
        op = op_module
    return op


def generate_video_from_frames(output_folder, temp_folder_name, frame_count, fps, platform):
    total = 0
    print("Video is generating...")
    filenames = sorted(glob.glob(os.path.join(output_folder, temp_folder_name, '*.jpg')))
    for filename in filenames:
        total += 1
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        if total == 1:
            if platform == "win32":
                video_path = os.path.join(output_folder, f"{temp_folder_name}_skeleton.mp4")
                out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            else:
                video_path = os.path.join(output_folder, f"{temp_folder_name}_skeleton.avi")
                out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MPEG'), fps, size)
        out.write(img)
        print(f"{total} / {frame_count}", end="\r")

    print(f"Finished skeleton video for file: {temp_folder_name}")
    out.release()

    if platform == "win32":
        cmd_ffmpeg = f"ffmpeg -i {os.path.join(output_folder, f'{temp_folder_name}_skeleton.mp4')} -vcodec h264 -acodec aac {os.path.join(output_folder, f'{temp_folder_name}_skeleton_compre.mp4')}"
    else:
        cmd_ffmpeg = f"ffmpeg -i {os.path.join(output_folder, f'{temp_folder_name}_skeleton.avi')} -vcodec h264 -acodec aac {os.path.join(output_folder, f'{temp_folder_name}_skeleton_compre.mp4')}"

    os.system(cmd_ffmpeg)
    os.remove(video_path)
    remove_folder_path = output_folder + '/' + temp_folder_name
    try:
        shutil.rmtree(remove_folder_path)
        print(f"Deleted folder: {remove_folder_path}")
    except Exception as e:
        print(f"Error deleting {remove_folder_path}: {e}")



def get_video_files(input_folder):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv']  # You can extend this list as needed
    all_files = os.listdir(input_folder)
    video_files = [f for f in all_files if any(f.endswith(ext) for ext in video_extensions)]
    return sorted(video_files)

def process_video_files(input_folder, output_folder, opWrapper, platform):
    content_input_folder = get_video_files(input_folder)
    print(f"Number of files to be processed: {len(content_input_folder)}")

    for name_of_file_and_exercise in tqdm(content_input_folder):
        path_to_video = os.path.join(input_folder, name_of_file_and_exercise)
        temp_folder_name = name_of_file_and_exercise[:-4]
        cam = cv2.VideoCapture(path_to_video)

        fps = cam.get(cv2.CAP_PROP_FPS)
        width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        print(f'fps = {fps}, Number of frames = {frame_count}, Duration (S) = {duration}')

        datum = op.Datum()
        ret, imageToProcess = cam.read()
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        body = np.full((frame_count, 25, 3), np.nan)

        os.makedirs(os.path.join(output_folder, temp_folder_name), exist_ok=True)

        for i in range(frame_count - 1):
            try:
                img = datum.cvOutputData.copy()
                filename = os.path.join(output_folder, temp_folder_name, f"{str(i).zfill(6)}_{temp_folder_name}.jpg")
            except Exception as e:
                    print(f"An error occurred: {e}")


            try:
                ret, imageToProcess = cam.read()
                datum.cvInputData = imageToProcess
                opWrapper.emplaceAndPop([datum])
                if datum.poseKeypoints is not None and datum.poseKeypoints.shape == (1, 25, 3):

                    body[i, :, :] = datum.poseKeypoints[0]
                    body[i][body[i] == 0] = np.nan
                img = datum.cvOutputData.copy()
                cv2.imwrite(filename, img)

            except Exception as e:
                print(f"An error occurred: {e}")
                #img = 255 * np.ones((int(height), int(width), 3), np.uint8)
                #img = imageToProcess.copy()
                # Scale the font size based on the image's dimensions
                # Here, we scale it based on the width; you can adjust as needed
                fontScale = width / 1000.0  # This is an arbitrary scaling factor; you might want to adjust it

                # Get the size of the text box
                text = 'No detected person in this frame'
                (textWidth, textHeight), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 4)

                # Compute starting position for the text so it's centered
                x = int((width - textWidth) // 2)
                y = int((height + textHeight) // 2)

                # Place the text on the image
                # Place the text on the image
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 2)
                cv2.imwrite(filename, img)
                print(f"Frame n.o.: {i} can't be processed.")

        k = np.array(body)
        print(f"Finished Openpose for file: {temp_folder_name}")

        scipy.io.savemat(os.path.join(output_folder, f"{temp_folder_name}.mat"),
                         mdict={'xbody': k[:, :, 0], 'ybody': k[:, :, 1], 'cbody': k[:, :, 2]})
        np.save(os.path.join(output_folder, f"{temp_folder_name}.npy"), k)

        df = create_dataframe(k, keypoint_names)
        df.to_csv(os.path.join(output_folder, f"{temp_folder_name}.csv"), index=False)
        # Saving as HDF5 (more space efficient and faster for large datasets)
        df.to_hdf(os.path.join(output_folder, f"{temp_folder_name}.h5"), key='keypoints', mode='w')

        generate_video_from_frames(output_folder, temp_folder_name, frame_count, fps, platform)

def main():
    args = get_arguments()
    input_folder_name = args.input_folder

    dir_path = os.path.dirname(os.path.realpath(__file__))

    try:

        op = setup_openpose(sys.platform, dir_path)  # Capture the returned reference
        if not op:  # Check if the setup was successful
            print('Error: Openpose Library, or Cuda is not installed. Exiting.')
            sys.exit(1)
    except ImportError:
        print('Error: Openpose Library, or Cuda is not installed. Exiting.')
        sys.exit(1)

    params = {
        "model_folder": "models/",
        #"number_people_max": 1,
        "face": False,
        "hand": False,
        "flir_camera": False
    }

    try:
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        print("OpenPose Library loaded successfully")
    except Exception as e:
        print(f'Error initializing OpenPose: {e}')
        sys.exit(1)

    if sys.platform == "win32":
        input_folder = os.path.join('C:\\', 'Users', 'adolfjin', 'testvideos')
        output_folder = os.path.join(input_folder, 'output')
    else:
        input_folder = f'/mnt/docker-openpose-hdd/{input_folder_name}'
        output_folder = f'/mnt/docker-openpose-hdd/{input_folder_name}_out'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder {output_folder} has been created.")
    else:
        print(f"Folder: {output_folder}, already exists.")

    process_video_files(input_folder, output_folder, opWrapper, sys.platform)


if __name__ == "__main__":
    main()