# WEyeDS v0 Reference

## Overview
WEyeDS v0 contains images from 54 participants xompleting a prosaccade task. 
Each participant looked at 100 dots in 50 different areas ion the screen.

This repository contains the code used to generate the dataset from the raw
data files. <code>process.py</code> was used to parse videos into frames and associate
the dot locations with the frames. <code>prepare.py</code> was used to get the eye corner
landmarks and prepare the final version of the dataset. <code>recalculate_y.py</code> is
a script to scale the y-coordinate to be on the same scale as x-coordinate (it was not rescaled
in the published dataset, but it was rescaled for model training). Finally, 
<code>serialize_to_tfrecord.py</code> was used to format the dataset for training.


## Data Structure
All participant IDs are random strings that do not have any assignment pattern. One ID is "undefined"
due to the ID transfer error, but the data from that participant are valid.

To get the dataset, please visit https://robbinslab.github.io/weyeds

The JSON file with the data information is structured as follows:

```bash
{participant_id : {
    meta : {
        "user_agent" # Browser identification
        "platform" # Operating system
        "screen_width" # Monitor width
        "screen_height" # Monitor height
        "scroll_width" # Width of the window content (px)
        "scroll_height" # Height of the window content (px)
        "window_inner_width" # Width of the browser viewport
        "window_inner_height" # Height of the browser viewport
        "device_pixel_ratio" # Scaling factor that maps virtual onto physical pixels
    },
    dot_info : {
        dot_id : {
            frame_id : {
                "file_name_right" # Name of the right eye crop image
                "file_name_left" # Name of the left eye crop image
                "file_name_full" # Name of the full-face image
                "right_landmarks" # Array [[X, Y], [X, Y]] Coordinates of the eye corner landmarks for the right eye
                "left_landmarks" # Array [[X, Y], [X, Y]] Coordinates of the eye corner landmarks for the left eye
                "label" # Array [X, Y] Dot location as a proportion of a viewport
            }
        }
    },
    split # train | valid | test
}
```