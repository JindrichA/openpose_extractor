
''' Created on 5.9.2023 - As a tool to extract body keypoints from standard mp4 video
    Tested on Windows and Linux  JindrichA (Jindrich Adolf)
'''
import sys
import os
from sys import platform
import argparse
import cv2
import numpy as np
import scipy, scipy.io
from tqdm import tqdm
import glob
import shutil


input_folder_name = "test"

dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        sys.path.append(dir_path + '/python/openpose/Release')
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/x64/Release;' + dir_path + '/bin;'
        import pyopenpose as op

        input_folder_name = "testovaci"
        input_folder = f'input/{input_folder_name}'
        output_folder = f'output/{input_folder_name}'
    else:
        # Linux import
        input_folder_name = '06092023Tst'

        sys.path.append('/mnt/docker-openpose/openpose/build/python')
        sys.path.append('/mnt/docker-openpose/openpose')
        output_folder = f'/mnt/docker-openpose-hdd/{input_folder_name}_out'
        input_folder = f'/mnt/docker-openpose-hdd/{input_folder_name}'
        from openpose import pyopenpose as op
except ImportError as e:
    print('Openpose Library, or Cuda is not installed')
    raise e

# OpenPose parameters, model choice etc
parser = argparse.ArgumentParser()
args = parser.parse_known_args()
params = dict()
params["model_folder"] = "models/"
params["number_people_max"]= 1
params["face"] = False
params["hand"] = False
params["flir_camera"] = False
params["inputfolder"] = input_folder_name

for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

try:
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    print("OpenPose Library loaded sucessfully")
except ImportError as e:
    print('Openpose Library, or Cuda is not installed')
    raise e

isFile = os.path.isdir(output_folder)
if isFile == False:
    os.mkdir(output_folder)
    print(f"Folder {input_folder_name} has been created.")
else:
    print(f"Folder: {input_folder_name}, already existed.")

list_dir = os.listdir(output_folder)
content_input_folder=sorted(os.listdir(input_folder))
print("Number of files to be processed: "+str(len(content_input_folder)))

for name_of_file_and_exercise in tqdm(content_input_folder):

    index_of_the_file = content_input_folder.index(name_of_file_and_exercise)
    celkem_souboru_ke_zpracovani = len(content_input_folder)
    try:
        path_to_video = input_folder + '/' + name_of_file_and_exercise
        temp_folder_name = name_of_file_and_exercise[:-4]
        cam = cv2.VideoCapture(path_to_video)
        fps = cam.get(cv2.CAP_PROP_FPS)
        width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        print('fps = ' + str(fps) + ', Number of frames = ' + str(frame_count) + ', Duration (S) = ' + str(duration))
        datum = op.Datum()
        ret, imageToProcess = cam.read()
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])
        length_of_the_input_video = frame_count
        body = np.zeros((frame_count, 25, 3))
        os.mkdir(output_folder + '/' + name_of_file_and_exercise[:-4])
        for i in (range(length_of_the_input_video - 1)):
            try:
                ret, imageToProcess = cam.read()
                datum.cvInputData = imageToProcess
                opWrapper.emplaceAndPop([datum])
                body[i, :, :] = (datum.poseKeypoints[0])
                # print(str(i) + ' / ' + str(frame_count), end="\r")
                img = datum.cvOutputData.copy()
                cv2.imwrite(
                    output_folder + '/' + temp_folder_name + '/' + str(i).zfill(6) + "_" + temp_folder_name + '.jpg',
                    img)
                # # k = cv2.waitKey(10)
                # if k == 27:
                #     # text_file_session.close()
                #     break
                # elif k == -1:
                #     continue
            except Exception as e:
                print(f"An error occurred: {e}")
                img = 255 * np.ones((int(height), int(width), 3), np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 500)
                fontScale = 1
                fontColor = (0, 0, 255)
                lineType = 2
                cv2.putText(img, 'Detection has failed for this frame!',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)
                cv2.imwrite(
                    output_folder + '/' + temp_folder_name + '/' + str(i).zfill(6) + "_" + temp_folder_name + '.jpg',
                    img)
                print("Frame n.o.: " + str(i) + " can't be processed.", end="\r")
        k = np.array(body)
        print("Finish Openpose for file : " + temp_folder_name)
        scipy.io.savemat(output_folder + '/' + temp_folder_name + '.mat',
                         mdict={'xbody': k[:, :, 0], 'ybody': k[:, :, 1], 'cbody': k[:, :, 2]})
        np.save(output_folder + '/' + temp_folder_name + '.npy', k)
        try:
            total = 0
            print("Video is generating...")
            for filename in sorted(glob.glob(output_folder + '/' + temp_folder_name + '/*.jpg')):
                total += 1
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width, height)
                if total == 1:
                    # out = cv2.VideoWriter(output_folder+'/'+temp_folder_name+'_skeleton.mp4',cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
                    if platform == "win32":
                        out = cv2.VideoWriter(output_folder + '/'+ temp_folder_name+"_skeleton.mp4",
                                              cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
                    else:
                        out = cv2.VideoWriter(output_folder + '/' + temp_folder_name + '_skeleton.avi',
                                              cv2.VideoWriter_fourcc(*'MPEG'), fps, size)
                out.write(img)
                print(str(total) + ' / ' + str(frame_count), end="\r")
            print("Finish skeleton video for file : " + temp_folder_name)
            out.release()
            if platform == "win32":
                puvodni_soubor = output_folder + '/' + temp_folder_name + '_skeleton.mp4'
                command_ffmpeg = "ffmpeg -i " + output_folder +'/'+temp_folder_name+'_skeleton.mp4' + " -vcodec h264 -acodec aac " + output_folder + '/' + temp_folder_name + '_skeleton_compre.mp4'
            else:
                puvodni_soubor = output_folder + '/' + temp_folder_name + '_skeleton.avi'
                command_ffmpeg = "ffmpeg -i " + output_folder + '/' + temp_folder_name + '_skeleton.avi' + " -vcodec h264 -acodec aac " + output_folder + '/' + temp_folder_name + '_skeleton_compre.mp4'
            try:
                os.system(command_ffmpeg)
                os.remove(puvodni_soubor)
            except:
                continue
            remove_folder_path = output_folder + '/' + temp_folder_name
            try:
                shutil.rmtree(remove_folder_path)
                print(f"Deleted folder: {remove_folder_path}")
            except Exception as e:
                print(f"Error deleting {remove_folder_path}: {e}")
        except:
            print("Cant genereate raw video")
    except:
        print("Doesn't work for: " + name_of_file_and_exercise)
print("Done for all videos sucessfully.")








