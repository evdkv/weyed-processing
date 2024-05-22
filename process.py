'''
File containing the main function 
to process the raw data.
'''

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import cv2, json, os, subprocess

def main():
    p_meta_dict = {}

    with open('dots/participant_meta.json', 'w') as p_file:
        with open("results/metadata.json", "r") as f:
            meta = json.load(f)
            for result in meta["data"][0]["studyResults"]:
                p_meta_dict.update(process_dataset(result))
        json.dump(p_meta_dict, p_file)

def process_dataset(result: dict) -> None:
    '''
    Process the dataset for a single result

    Parameters:
        result (dict): The result dict from metadata to process
    '''
    p_meta_dict = {}
    result_id = result["id"]

    try:
        participant_id = result["urlQueryParameters"]["participant_id"]
    except KeyError:
        participant_id = "undefined"
    print(f"Participant {participant_id} with result {result_id} is being processed")
    
    # Retrieve the path to the result data
    path = 'results/' + result["componentResults"][0]["path"]

    # Stitch the video chunks and get a filename for the new video
    filename = stitch_recording(path, participant_id)

    # Convert the video to an uncorrupted WEBM file
    convert_video(filename)

    # Get the count and time stamps for the black dots
    bdots, bdot_count = get_bdots(path, result_id, participant_id)

    # Get the video chunks for the black dots
    map_bdots(participant_id, bdots, f"full_videos/{filename}.webm")

    # Make frames for the black dots
    frame_count = make_frames(bdots, participant_id)

    # Update the participant metadata dictionary
    p_meta_dict[participant_id] = {"dot_count" : bdot_count, "frame_count": frame_count}

    return p_meta_dict

def make_frames(bdots: dict, pid: int) -> int:
    '''
    Extract frames from the black dot video chunks
    
    Parameters:
        bdots (dict): The dictionary of black dots
        pid (int): The participant ID
    
    Returns:
        int: The new frame count
    '''
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

        # Extract the frames in the latter part of the chunk
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
    '''
    Extract the video chunks for the black dot timestamps

    Parameters:
        pid (int): The participant ID
        bdots (dict): The dictionary of black dots
        full_video_name (str): The name of the full video
    '''
    print(f"Mapping bdots for participant {pid}")
    for bdot in bdots:
        ffmpeg_extract_subclip(full_video_name, bdots[bdot]["time_run"] / 1000, bdots[bdot]["time_end"] / 1000,
                                targetname=f"dots/{pid}/bdot_{bdot}.webm")


def convert_video(filename: str) -> None:
    '''
    Convert the video to an uncorrupted WEBM format

    Parameters:
        filename (str): The name of the file to convert
    '''
    print(f"Converting {filename}.webm")
    subprocess.run(["towebm", "--delete-log", f"full_videos/{filename}.webm"])
    os.remove(f"full_videos/{filename}.webm")
    os.rename(f"{filename}.webm", f"full_videos/{filename}.webm")

def stitch_recording(path: str, pid: int) -> str:
    '''
    Read in the videos in the binary 
    format and stitch them together

    Parameters:
        path (str): The path to the results
        pid (int): The participant ID

    Returns:
        str: The filename of the new video
    '''
    i = 0
    inp = b''
    while True:
        try:
            inp1 = open(f'{path}/files/{pid}_video_{i}.webm', 'rb').read()
            inp += inp1
            i += 1
        except FileNotFoundError:
            break

    filename = f'{pid}'

    with open(f'full_videos/{filename}.webm', 'wb') as fp:
        fp.write(inp)

    return filename

def get_bdots(path: str, result_id: int, pid: int) -> tuple[dict, int]:
    '''
    Get the black dot timestamps and coordinates

    Parameters:
        path (str): The path to the results
        result_id (int): The result ID
        pid (int): The participant ID
    
    Returns:
        tuple[dict, int]: The dictionary of black dots and the count
    '''
    with open(f"results/{path}/data.txt", "r") as f: 
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