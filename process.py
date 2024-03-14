from moviepy.editor import VideoFileClip
import cv2, json, os, requests, subprocess


def retrieve_data():
    headers = {
        'accept': 'application/json',
        'Authorization': 'Bearer ' + os.getenv('JATOS_API_TOKEN'),
    }

    params = {
        "studyId": "1",
        "studyResultId": "40",
    }

    response = requests.post('https://jatos.robbins-lab.com/jatos/api/v1/results', headers=headers, params=params)

    with open('40.jrzip', 'wb') as f:
        f.write(response.content)

def main():

    p_meta_dict = {}

    with open('dots/participant_meta.json', 'w') as p_file:
        # Get the metadata from results
        with open("results/metadata.json", "r") as f:
            meta = json.load(f)
            for result in meta["data"][0]["studyResults"]:
                p_meta_dict.update(process_dataset(result))
        json.dump(p_meta_dict, p_file)

def process_dataset(result: dict) -> None:
    p_meta_dict = {}
    # Process files for each result
    result_id = result["id"]
    participant_id = result["urlQueryParameters"]["participant_id"]
    print(f"Participant {participant_id} with result {result_id} is being processed")
    
    # Get the dict of bdot data
    filename = stitch_recording(result_id, participant_id)
    convert_video(filename)

    bdots, bdot_count = get_bdots(result_id, participant_id)
    map_bdots(participant_id, bdots, f"full_videos/{filename}.webm")
    frame_count = make_frames(bdots, participant_id)

    p_meta_dict[participant_id] = {"dot_count" : bdot_count, "frame_count": frame_count}

    return p_meta_dict

def make_frames(bdots: dict, pid: int) -> int:
    print(f"Making frames for participant {pid}")
    for bdot in bdots:
        vidcap = cv2.VideoCapture(f"dots/{pid}/bdot_{bdot}.webm")
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(f"dots/{pid}/{bdot}_frame_{count}.jpg", image)
            success, image = vidcap.read()
            count += 1
        os.remove(f"dots/{pid}/bdot_{bdot}.webm")

        # Now remove the invalid frames

        valid_range = (count - 10, count - 5)
        new_count = 0
        for i in range(0, count):
            if i < valid_range[0] or i > valid_range[1]:
                print(f"Removing frame {i}")
                os.remove(f"dots/{pid}/{bdot}_frame_{i}.jpg")
            else:
                print(f"Renaming frame {i} to {new_count}")
                os.rename(f"dots/{pid}/{bdot}_frame_{i}.jpg", f"dots/{pid}/{bdot}_frame_{new_count}.jpg")
                new_count += 1

    return new_count
        

def map_bdots(pid: int, bdots: dict, full_video_name: str) -> None:
    print(f"Mapping bdots for participant {pid}")
    video = VideoFileClip(full_video_name)
    for bdot in bdots:
        clip = video.subclip(bdots[bdot]["time_run"] / 1000, bdots[bdot]["time_end"] / 1000)
        clip.write_videofile(f"dots/{pid}/bdot_{bdot}.webm")

def convert_video(filename: str) -> None:
    print(f"Converting {filename}.webm")
    subprocess.run(["towebm", "--delete-log", f"full_videos/{filename}.webm"])
    os.remove(f"full_videos/{filename}.webm")
    os.rename(f"{filename}.webm", f"full_videos/{filename}.webm")

def stitch_recording(rid: int, pid: int) -> str:
    i = 0
    inp = b''
    while True:
        try:
            inp1 = open(f'results/study_result_{rid}/comp-result_{rid}/files/{pid}_video_{i}.webm', 'rb').read()
            inp += inp1
            i += 1
        except FileNotFoundError:
            break

    filename = f'{pid}'

    with open(f'full_videos/{filename}.webm', 'wb') as fp:
        fp.write(inp)

    return filename

def get_bdots(result_id: int, pid: int) -> tuple[dict, int]:
    with open(f"results/study_result_{result_id}/comp-result_{result_id}/data.txt", "r") as f: 
        data = json.load(f)
        row = 0
        bdot_count = 0
        bdot_dict = {}
        while True:
            try:
                if data[row]["sender"] == "i4" and data[row]["rec_state_ch"] == "recording":
                    video_pad_on = data[row]["rec_state_ch_stamp"]

                elif data[row]["sender"] == "i5" and data[row]["rec_state_ch"] == "paused":
                    video_pad_off = data[row]["rec_state_ch_stamp"]
                    video_pad = video_pad_off - video_pad_on

                elif data[row]["sender"] == "bdot_canvas":
                    try:
                        if data[row]["rec_state_ch"] == "recording":
                            camera_on = data[row]["rec_state_ch_stamp"]
                    except KeyError:
                        pass

                    time_run = (data[row]["time_run"] - camera_on) + video_pad
                    time_end = (data[row]["time_end"] - camera_on) + video_pad

                    bdot_dict[bdot_count] = {"result_id": result_id, "participant_id": pid, 
                                             "coords": data[row]["coords"], "time_run": time_run, 
                                             "time_end": time_end}
                    bdot_count += 1
                row += 1
            except IndexError:
                break
    os.mkdir(f"dots/{pid}")
    with open(f"dots/{pid}/dots_meta.json", "w") as f:
        json.dump(bdot_dict, f)

    return bdot_dict, bdot_count

if __name__ == '__main__':
    main()