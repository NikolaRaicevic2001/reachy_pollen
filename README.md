# Reachy_Pollen
Reachy_Pollen is a Python-based control and experimentation framework for the Reachy robot, built on top of the official SDK. It provides modular tools for robot manipulation, teleoperation integration, and rapid prototyping of control algorithms, enabling streamlined development of perception, planning, and interaction pipelines.

Documentation: https://pollen-robotics.github.io/reachy2-sdk/reachy2_sdk/reachy_sdk.html

# Setup
## Docker
Image Dowload: https://hub.docker.com/r/pollenrobotics/reachy2

General Docker
```
docker run --rm --platform linux/amd64 -p 8888:8888 -p 6080:6080 -p 50051:50051 --name reachy2 docker.io/pollenrobotics/reachy2
```

Gazebo Docker
```
docker run --rm --platform linux/amd64 -p 8888:8888 -p 6080:6080 -p 50051:50051 --name reachy2 docker.io/pollenrobotics/reachy2 start_rviz:=true start_sdk_server:=true fake:=true orbbec:=false gazebo:=true
```

Mujoco Docker
```
docker run --rm --platform linux/amd64 -p 8888:8888 -p 6080:6080 -p 50051:50051 --name reachy2 docker.io/pollenrobotics/reachy2 start_rviz:=true start_sdk_server:=true fake:=true orbbec:=false mujoco:=true
```

Different Scene Docker
```
docker run --rm --platform linux/amd64 -p 8888:8888 -p 6080:6080 -p 50051:50051 --name reachy2 docker.io/pollenrobotics/reachy2 start_rviz:=true start_sdk_server:=true fake:=true orbbec:=false mujoco:=true scene:=fruits
```

## Virtual Envrionment
Setup virtual environment
- Linux
```
python3.10 -m venv reachy2
source reachy2/bin/activate
```
- Windows
```
py -3.10 -m venv reachy2
.\reachy2\Scripts\Activate
```

Install requirments
```
pip install -r requirements.txt
```

## Visualization
Rviz: http://localhost:6080/vnc.html?autoconnect=1&resize=remote%e2%81%a0

# Teleoperation
## Setup
1) Dowload the VR headset app: https://www.meta.com/quest/setup/?srsltid=AfmBOorKOuGUIU7NR95vBQ4dcVi464ir4qGZndC4WYzo4wcg1Jpg4bKb
    - Connect the VR headset to the local machine 
    - Make sure that you can display the computer screen in the VR headset world

2) Install the Pollen Reachy latest VR application: https://github.com/pollen-robotics/Reachy2Teleoperation/releases/latest/download/Reachy2Teleoperation_installer.exe
    - Select the Complete installation for gstreamer
    - Launch the application Reachy2Teleoperation from your computer
    - Connect to the robot by setting the `robot name` = {anything} and `robot IP` = {either name `“r2-0008.local”` or IP `192.168.10.172`} 
    - Try connecting to reachy and verify if you see
        - a green text telling you “Connected to Reachy”
        - the view of the robot displayed in miniature
        - a good network connection indication

3) If the "Connection Failed" once entering the transition room with "motor","audio", and other configurations not being connected then:
    - Need to install gstreamer correctly: https://gstreamer.freedesktop.org/data/pkg/windows/1.27.90/msvc/?__goaway_challenge=meta-refresh&__goaway_id=9cf305064589a1220ab9c6f4cbaec4b1&__goaway_referer=https%3A%2F%2Fforum.pollen-robotics.com%2F
        - gstreamer-1.0-msvc-x86_64-1.27.90.msi	        2026-01-08 01:17 	89M	
        - gstreamer-1.0-devel-msvc-x86_64-1.27.90.msi	2026-01-08 01:13 	317M	 
    - After that we need to setup windows path environmental variables: 
        ```
        # Replace this path with your actual installation path
        $gstreamerPath = "C:\gstreamer\1.0\msvc_x86_64\bin"

        # Add it to current session PATH
        $Env:PATH += ";$gstreamerPath"

        # Optional: permanently add to user PATH
        [Environment]::SetEnvironmentVariable("PATH", $Env:PATH, "User")
        ```
    - Restart PowerShell and verify if the gstreamer is correctly setup: 
        ```
        gst-inspect-1.0 --version
        ```
    
4) If experiencing issues with VR headset not properly moving the arms and not reflecting that in the teleoperation app then:
    - Set the VR flag environmental variable: 
        ```
        $env:UNITY_XR_ENABLE=1
        ```
## Dataset Recording
### Recording
```
lerobot-record --robot.type=reachy2 --robot.ip_address=192.168.10.172 --robot.id=r2-0008 --robot.use_external_commands=true --teleop.type=keyboard --robot.with_mobile_base=false --robot.with_torso_camera=false --dataset.repo_id=pollen_robotics/record_test --dataset.single_task="Reachy 2 recording test" --dataset.num_episodes=1 --dataset.episode_time_s=5 --dataset.fps=15 --dataset.push_to_hub=false --dataset.private=true --dataset.streaming_encoding=true --dataset.encoder_threads=2 --display_data=true
```

```
lerobot-record `
--robot.type=reachy2 `
--robot.ip_address=192.168.137.162 `
--robot.id=r2-0008 `
--robot.use_external_commands=true `
--robot.with_mobile_base=true `
--robot.with_l_arm=true `
--robot.with_r_arm=true `
--robot.with_neck=true `
--robot.with_antennas=true `
--robot.with_left_teleop_camera=false `
--robot.with_right_teleop_camera=false `
--robot.with_torso_camera=false `
--robot.camera_width=640 `
--robot.camera_height=480 `
--robot.disable_torque_on_disconnect=false `
--robot.max_relative_target=5.0 `
--teleop.type=reachy2_teleoperator `
--teleop.ip_address=192.168.137.162 `
--teleop.use_present_position=true `
--teleop.with_mobile_base=true `
--teleop.with_l_arm=true `
--teleop.with_r_arm=true `
--teleop.with_neck=true `
--teleop.with_antennas=true `
--dataset.repo_id=pollen_robotics/record_test `
--dataset.single_task="Reachy 2 recording test" `
--dataset.num_episodes=1 `
--dataset.episode_time_s=10 `
--dataset.fps=30 `
--dataset.push_to_hub=false `
--dataset.private=true `
--dataset.streaming_encoding=false `
--dataset.encoder_threads=8 `
--display_data=false
```

- Higher Frequency Setup
```
lerobot-record `
--robot.type=reachy2 `
--robot.ip_address=192.168.137.162 `
--robot.id=r2-0008 `
--robot.use_external_commands=true `
--robot.with_mobile_base=false `
--robot.with_l_arm=true `
--robot.with_r_arm=true `
--robot.with_neck=true `
--robot.with_antennas=false `
--robot.with_torso_camera=false `
--robot.camera_width=640 `
--robot.camera_height=480 `
--teleop.type=reachy2_teleoperator `
--teleop.ip_address=192.168.137.162 `
--teleop.use_present_position=true `
--teleop.with_mobile_base=false `
--teleop.with_l_arm=true `
--teleop.with_r_arm=true `
--teleop.with_neck=true `
--teleop.with_antennas=false `
--dataset.repo_id=erl-hub/reachy-pick-and-place `
--dataset.single_task="Reachy 2 pick and place test" `
--dataset.num_episodes=1 `
--dataset.episode_time_s=30 `
--dataset.fps=15 `
--dataset.vcodec=h264_nvenc `
--dataset.streaming_encoding=false `
--dataset.push_to_hub=true `
--display_data=false `
--resume=true
```

- Remove or rename dataset after every recording:
```
Remove-Item -Recurse -Force "C:\Users\nikra\.cache\huggingface\lerobot\erl-hub\reachy-pick-and-place"
Remove-Item -Recurse -Force "C:\Users\nikra\.cache\huggingface\lerobot\erl-hub\reachy-pick-and-place-images"
```

### Upload the datasets
```
# Navigate to the data
cd "C:\Users\nikra\.cache\huggingface\lerobot\erl-hub\reachy-pick-and-place"

# Use the dedicated CLI uploader (often bypasses API timeouts)
huggingface-cli upload erl-hub/reachy-pick-and-place . . --repo-type=dataset
```

### Profiling
```
Get-Process | Where-Object {$_.ProcessName -like "*python*"}
```

```
py-spy record -o lerobot_profile.svg --pid 24016
```

### Replaying
- Locally
```
lerobot-replay `
    --robot.type=reachy2 `
    --robot.ip_address=192.168.137.162 `
    --robot.use_external_commands=false `
    --robot.with_mobile_base=false `
    --dataset.repo_id=erl-hub/reachy-pick-and-place/ `
    --dataset.episode=2
```

- Hub
```
lerobot-replay `
--robot.type=reachy2 `
--robot.ip_address=192.168.137.162 `
--robot.id=r2-0008 `
--robot.use_external_commands=false `
--robot.with_mobile_base=false `
--robot.with_l_arm=true `
--robot.with_r_arm=true `
--robot.with_neck=true `
--robot.with_antennas=false `
--dataset.repo_id="erl-hub/reachy-pick-and-place" `
--dataset.episode=4
```