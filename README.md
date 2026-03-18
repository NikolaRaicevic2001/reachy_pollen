# Reachy_Pollen
Reachy_Pollen is a Python-based control and experimentation framework for the Reachy robot, built on top of the official SDK. It provides modular tools for robot manipulation, teleoperation integration, and rapid prototyping of control algorithms, enabling streamlined development of perception, planning, and interaction pipelines.

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
```
python3.10 -m venv reachy2_env
source reachy2_env/bin/activate
```
Install requirments
```
pip install -r requirements.txt
```

## Visualization
Rviz: http://localhost:6080/vnc.html?autoconnect=1&resize=remote%e2%81%a0

# Teleoperation Section
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
    - Set the VR environmental variable: 
        ```
        $env:UNITY_XR_ENABLE=1
        ```