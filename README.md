# CARRASS

**Camera and Radar Robotic Arm Safety System**

>
> This is a WIP project, do not expect any working code or meaningful documentation
>


## Dev Setup

1. Download recorded session:
https://tum.fredhennecke.com/x/57e09ad469fab426/82276a69ad7dec5e.zip

2. Extract the zip file into `recordings`

3. Either just replay a recorded session:
    ```
    python -m utils.capture_replay_visualize --mode play --name session_person
    ```
    or run the main system with a recorded session as input: 
    ```
    python main.py --mode play --name session_person --verbose
    ```
   
## Components

| Component | Function | Visualization                             |
| :--- | :--- |:------------------------------------------|
| **Kalman Filter** | Performs **Sensor Fusion** to provide optimal state estimates by filtering out noise from multiple data sources. | <img src="docs/kalman.png" width="100%"> |

