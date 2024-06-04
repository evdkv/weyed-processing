'''
File containing the script to prepare
the data for model training: generating eye corner
landmarks, eye crops and saving them.
'''

import cv2
import mediapipe as mp
import math
import json, os
import shutil

def main(split: list):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    info_dict = {}
    cur_split = split.pop(0)
    os.mkdir('processed')

    with open('dots/participant_meta.json', 'r') as p_file:
        p_meta_dict = json.load(p_file)
        for pid in p_meta_dict:

            # No faces detected in this participant
            if pid == 80170:
                continue

            with open(f'dots/{pid}/dots_meta.json', 'r') as f:
                dot_dict = json.load(f)

            info_dict[pid] = {}

            print(f"Processing participant {pid}", flush=True)
            if cur_split[0] == 0:
                cur_split = split.pop(0)

            bdot_dict = {}
            for bdot in range(0, p_meta_dict[pid]["dot_count"]):
                

                print(f"Processing bdot {bdot}", flush=True)
                frame_info = {}
                for frame in range(0, p_meta_dict[pid]["frame_count"]):
                    
                    print(f"Processing frame {frame}", flush=True)
                    eye_crop_right, eye_crop_left, right_landmarks, left_landmarks = get_crops_landmarks(f"dots/{pid}/{bdot}_frame_{frame}.jpg", face_mesh)

                    # Ensure the crops are valid
                    if eye_crop_right is None or eye_crop_left is None:
                        continue
                    
                    # Ensure the crops are of the correct size
                    if eye_crop_right.shape[0] != 128 or eye_crop_right.shape[1] != 128 or eye_crop_left.shape[0] != 128 or eye_crop_left.shape[1] != 128:
                        continue

                    # Write the cropped eye images
                    cv2.imwrite(f"processed/{pid}_{bdot}_{frame}_right.jpg", eye_crop_right)
                    cv2.imwrite(f"processed/{pid}_{bdot}_{frame}_left.jpg", eye_crop_left)

                    shutil.copy(src=f"dots/{pid}/{bdot}_frame_{frame}.jpg", dst=f"processed/{pid}_{bdot}_{frame}_full.jpg")
  
                    img_info = {"file_name_right": f"{pid}_{bdot}_{frame}_right.jpg", 
                                "file_name_left": f"{pid}_{bdot}_{frame}_left.jpg", 
                                "file_name_full" : f"{pid}_{bdot}_{frame}_full.jpg",
                                "right_landmarks": right_landmarks, 
                                "left_landmarks": left_landmarks, 
                                "label": [dot_dict[str(bdot)]["coords"]["X"], dot_dict[str(bdot)]["coords"]["Y"]]}

                    frame_info.update({frame : img_info})
                    
                    print(f"Processed frame {frame} for bdot {bdot} for participant {pid}", flush=True)
                bdot_dict.update({bdot : frame_info})

            participant_meta_dict = {}

            # Get metadata from results
            with open("results/metadata.json", "r") as mdata:
                metadata = json.load(mdata)
                for result in metadata["data"][0]["studyResults"]:
                    try:
                        test_pid = result["urlQueryParameters"]["participant_id"]
                    except KeyError:
                        test_pid = "undefined"
                    if result["studyState"] == "FINISHED" and str(pid) == test_pid:
                        with open(f"{'results' + result['componentResults'][0]['path']}/data.txt") as resdata:
                            data = json.load(resdata)
                            participant_meta_dict.update({"user_agent": data[0]["meta"]["userAgent"],
                                                            "platform": data[0]["meta"]["platform"],
                                                            "screen_width": data[0]["meta"]["screen_width"],
                                                            "screen_height": data[0]["meta"]["screen_height"],
                                                            "scroll_width": data[0]["meta"]["scroll_width"],
                                                            "scroll_height": data[0]["meta"]["scroll_height"],
                                                            "window_inner_width": data[0]["meta"]["window_innerWidth"],
                                                            "window_inner_height": data[0]["meta"]["window_innerHeight"],
                                                            "device_pixel_ratio": data[0]["meta"]["devicePixelRatio"]})
            # Append the metadata to the dictionary
            info_dict[pid].update({"meta" : participant_meta_dict, "dot_info" : bdot_dict, "split" : cur_split[1]})

            cur_split[0] = cur_split[0] - 1
    
    with open('participant_data.json', 'w') as f:
        json.dump(info_dict, f)


def get_crops_landmarks(img_path: str, face_mesh: mp.solutions.face_mesh.FaceMesh = None):
    '''
    Generate eye crops and landmarks for the eyes. Also, recalculate landmark coordinates
    to match the new eye crops.

    Parameters:
        img_path (str): The path to the image
        face_mesh (mp.solutions.face_mesh.FaceMesh): The MediaPipe Face Mesh object
    
    Returns:
        tuple: The right eye crop, left eye crop, right eye landmarks, left eye landmarks
    '''

    # Process the image
    image = cv2.imread(img_path)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Assuming faces are detected, extract the landmarks for the eyes
    # and draw them (Omitted for simplicity)

    detection = results.multi_face_landmarks
    if detection is None:
        print(f"No face detected in {img_path}", flush=True)
        return None, None, None, None
    for i, landmark in enumerate(detection[0].landmark):
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])

        # get the corner landmarks of the eye
        if i in [33, 133, 362, 263]:
            if i == 33:
                rl = (x, y)
            elif i == 133:
                rr = (x, y)
            elif i == 362:
                ll = (x, y)
            elif i == 263:
                lr = (x, y)
        # get top and bottom landmarks of the eye
        if i in [159, 145, 386, 374]:
            if i == 159:
                rt = (x, y)
            elif i == 145:
                rb = (x, y)
            elif i == 386:
                lt = (x, y)
            elif i == 374:
                lb = (x, y)
        
    right_pad = float(128 - (rr[0] - rl[0])) / 2
    left_pad = float(128 - (lr[0] - ll[0])) / 2
    top_pad = float(128 - (rb[1] - rt[1])) / 2
    bottom_pad = float(128 - (lb[1] - lt[1])) / 2


    r0 = rr[0] + math.floor(right_pad)
    r1 = rl[0] - math.ceil(right_pad)
    l0 = lr[0] + math.floor(left_pad)
    l1 = ll[0] - math.ceil(left_pad)
    rt0 = rt[1] - math.floor(top_pad)
    rb1 = rb[1] + math.ceil(bottom_pad)
    lt0 = lt[1] - math.floor(top_pad)
    lb1 = lb[1] + math.ceil(bottom_pad)

    # Recalculate landmarks
    rr = [rr[0] - r1, rr[1] - rt0]
    rl = [rl[0] - r1, rl[1] - rt0]

    lr = [lr[0] - l1, lr[1] - lt0]
    ll = [ll[0] - l1, ll[1] - lt0]

    rt0 -= 128 - (rb1 - rt0)
    lt0 -= 128 - (lb1 - lt0)

    r1 -= 128 - (r0 - r1)
    l1 -= 128 - (l0 - l1)

    eye_crop_right = image[rt0:rb1, r1:r0]
    eye_crop_left = image[lt0:lb1, l1:l0]

    return eye_crop_right, eye_crop_left, [rr, rl], [lr, ll]

if __name__ == "__main__":
    main([[42, "train"], [6, "valid"], [10, "test"]])