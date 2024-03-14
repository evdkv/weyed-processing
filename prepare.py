"""
What I need to do:
1. Read the dot frames
2. Determine eye corner landmarks
3. Determine the eye boxes
4. Make an eye crop
5. Assign training/validation/test split
6. Put into a TFRecord file along with the label and landmarks
"""

import cv2
import mediapipe as mp
import math
import json, os

def main(split: list):
        # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    metadict = {}
    cur_split = split.pop(0)
    os.mkdir('processed')
    with open('dots/participant_meta.json', 'r') as p_file:
        p_meta_dict = json.load(p_file)
        for pid in p_meta_dict:
            with open(f'dots/{pid}/dots_meta.json', 'r') as f:
                dot_dict = json.load(f)
            metadict[pid] = []
            print(f"Processing participant {pid}")
            if cur_split[0] == 0:
                cur_split = split.pop(0)
            for bdot in range(0, p_meta_dict[pid]["dot_count"]):
                print(f"Processing bdot {bdot}")
                for frame in range(0, p_meta_dict[pid]["frame_count"]):
                    print(f"Processing frame {frame}")
                    eye_crop_right, eye_crop_left, right_landmarks, left_landmarks = get_crops_landmarks(f"dots/{pid}/{bdot}_frame_{frame}.jpg", face_mesh)
                    cv2.imwrite(f"processed/{pid}_{bdot}_{frame}_right.jpg", eye_crop_right)
                    cv2.imwrite(f"processed/{pid}_{bdot}_{frame}_left.jpg", eye_crop_left)
                    metadict[pid].append({"file_name_right": f"{pid}_{bdot}_{frame}_right.jpg", "file_name_left": f"{pid}_{bdot}_{frame}_left.jpg",
                                    "right_landmarks": right_landmarks, "left_landmarks": left_landmarks, "label": [dot_dict[str(bdot)]["coords"]["X"], dot_dict[str(bdot)]["coords"]["Y"]],
                                    "split": cur_split[1]})
                    print(f"Processed frame {frame} for bdot {bdot} for participant {pid}")
            cur_split[0] = cur_split[0] - 1
    
    with open('processed/info.json', 'w') as f:
        json.dump(metadict, f)


def get_crops_landmarks(img_path: str, face_mesh: mp.solutions.face_mesh.FaceMesh = None):

    # Process the image
    image = cv2.imread(img_path)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Assuming faces are detected, extract the landmarks for the eyes
    # and draw them (Omitted for simplicity)

    detection = results.multi_face_landmarks[0]
    for i, landmark in enumerate(detection.landmark):
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


    eye_crop_right = image[rt0:rb1, r1:r0]
    eye_crop_left = image[lt0:lb1, l1:l0]

    return eye_crop_right, eye_crop_left, [rr, rl], [ll, lr]



if __name__ == "__main__":
    main([[2, "train"], [1, "valid"], [1, "test"]])
   # main([[29, "train"], [3, "valid"], [4, "test"]])