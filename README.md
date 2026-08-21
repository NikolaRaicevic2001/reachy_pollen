# Reachy_Pollen
Reachy_Pollen is a Python-based control and experimentation framework for the Reachy robot, built on top of the official SDK. It provides modular tools for robot manipulation, teleoperation integration, and rapid prototyping of control algorithms, enabling streamlined development of perception, planning, and interaction pipelines.

# Setup
## Docker: Reachy Image
Image Dowload: [https://hub.docker.com/r/pollenrobotics/reachy2](https://hub.docker.com/r/pollenrobotics/reachy2)

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

## Docker
```
docker pull nikolaraicevic2001/erl-lerobot:latest
```

## Virtual Envrionment
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

- Install requirments
```
pip install -r requirements.txt
```

## System Dependencies:
```
sudo apt update && sudo apt upgrade -y
sudo add-apt-repository ppa:ubuntuhandbook1/ffmpeg7
sudo apt update
sudo apt install ffmpeg -y
```
-Note: not required for Docker image, it's already included.

# Teleoperation
## Setup
1. Dowload the VR headset app: https://www.meta.com/quest/setup/?srsltid=AfmBOorKOuGUIU7NR95vBQ4dcVi464ir4qGZndC4WYzo4wcg1Jpg4bKb
  - Connect the VR headset to the local machine 
  - Make sure that you can display the computer screen in the VR headset world
2. Install the Pollen Reachy latest VR application: https://github.com/pollen-robotics/Reachy2Teleoperation/releases/latest/download/Reachy2Teleoperation_installer.exe
  - Select the Complete installation for gstreamer
  - Launch the application Reachy2Teleoperation from your computer
  - Connect to the robot by setting the `robot name` = {anything} and `robot IP` = {either name `“r2-0008.local”` or IP `192.168.10.172`} 
  - Try connecting to reachy and verify if you see
    - a green text telling you “Connected to Reachy”
    - the view of the robot displayed in miniature
    - a good network connection indication
3. If the "Connection Failed" once entering the transition room with "motor","audio", and other configurations not being connected then:
  - Need to install gstreamer correctly: [https://gstreamer.freedesktop.org/data/pkg/windows/]
  - Choose version 1.26.11
    - gstreamer-1.0-msvc-x86_64-1.26.11.msi              
    - gstreamer-1.0-devel-msvc-x86_64-1.26.11.msi    
  - After that we need to setup windows path environmental variables:
  - Restart PowerShell and verify if the gstreamer is correctly setup: 
4. If experiencing issues with VR headset not properly moving the arms and not reflecting that in the teleoperation app then:
  - Set the VR flag environmental variable: 

## Dataset Recording
### Recording
- Windows
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
--robot.with_left_teleop_camera=false `
--robot.with_right_teleop_camera=false `
--robot.with_torso_camera=true `
--robot.camera_width=640 `
--robot.camera_height=480 `
--robot.disable_torque_on_disconnect=false `
--robot.max_relative_target=5.0 `
--teleop.type=reachy2_teleoperator `
--teleop.ip_address=192.168.137.162 `
--teleop.use_present_position=true `
--teleop.with_mobile_base=false `
--teleop.with_l_arm=true `
--teleop.with_r_arm=true `
--teleop.with_neck=true `
--teleop.with_antennas=false `
--dataset.repo_id=erl-hub/reachy-pick-and-place-images `
--dataset.single_task="Pick up the tomato soup can and place it into the metal bowl" `
--dataset.num_episodes=1 `
--dataset.episode_time_s=30 `
--dataset.fps=15 `
--dataset.vcodec=h264 `
--dataset.streaming_encoding=false `
--dataset.private=false `
--dataset.push_to_hub=true `
--display_data=false `
--resume=true
```

- Remove dataset if you want to restart a recording:
```
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\lerobot\erl-hub\reachy-pick-and-place-images"
```

### Recording: On the Robot with Docker

#### One-time setup

Copy the patched Reachy2 interface to `bedrock`:

```powershell
scp .\teleoperation\robot_reachy2.py `
  bedrock@192.168.137.100:/home/bedrock/robot_reachy2.py
```

SSH into `bedrock`:

```powershell
ssh bedrock@192.168.137.100
```

Store the Hugging Face token:

```bash
mkdir -p ~/.config/lerobot
echo 'HF_TOKEN=hf_your_token_here' > ~/.config/lerobot/hf.env
chmod 600 ~/.config/lerobot/hf.env
```

#### Start LeRobot

```bash
docker run --rm -it \
  --network host \
  --env-file ~/.config/lerobot/hf.env \
  -v /home/bedrock/robot_reachy2.py:/lerobot/src/lerobot/robots/reachy2/robot_reachy2.py \
  -v /home/bedrock/lerobot-data:/data \
  huggingface/lerobot-cpu:latest \
  bash
```

If restarting the same dataset with `--resume=false`:

```bash
rm -rf ~/.cache/huggingface/lerobot/erl-hub/reachy-cleaning
```

Record:

```bash
lerobot-record \
  --robot.type=reachy2 \
  --robot.ip_address=127.0.0.1 \
  --robot.with_mobile_base=false \
  --robot.with_l_arm=true \
  --robot.with_r_arm=true \
  --robot.with_neck=true \
  --robot.with_antennas=false \
  --robot.with_left_teleop_camera=false \
  --robot.with_right_teleop_camera=false \
  --robot.with_torso_camera=true \
  --teleop.type=reachy2_teleoperator \
  --teleop.ip_address=127.0.0.1 \
  --teleop.with_mobile_base=false \
  --teleop.with_l_arm=true \
  --teleop.with_r_arm=true \
  --teleop.with_neck=true \
  --teleop.with_antennas=false \
  --dataset.repo_id=erl-hub/reachy-cleaning \
  --dataset.root=/data/reachy-cleaning \
  --dataset.single_task="Organize the table by placing the fruits in the bowl and cans, bottles, and boxes in the basket" \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=90 \
  --dataset.reset_time_s=60 \
  --dataset.fps=15 \
  --dataset.private=false \
  --dataset.push_to_hub=true \
  --display_data=false \
  --play_sounds=false \
  --resume=false
```
Use `--resume=false` for the first episode. For subsequent episodes, use:

```bash
--resume=true
```

### Upload the datasets
```
# Navigate to the data
cd "$env:USERPROFILE\.cache\huggingface\lerobot\erl-hub\reachy-pick-and-place"

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
    --robot.id=r2-0008 `
    --robot.use_external_commands=false `
    --robot.with_mobile_base=false `
    --robot.with_l_arm=true `
    --robot.with_r_arm=true `
    --robot.with_neck=true `
    --dataset.repo_id=erl-hub/reachy-pick-and-place-test `
    --dataset.root=outputs\reachy_local_test `
    --dataset.episode=0
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

## Training policies:
### Training locally
#### Convert dataset to required format
```
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=pollen-robotics/pick_and_place_bottle && \
```

#### Dowloading Datasets Locally (Optional)
```
hf download erl-hub/reachy-pick-and-place-images --repo-type dataset --local-dir /home/user_lerobot/erl_hub
hf download ganatrask/NOVA --repo-type dataset --local-dir /home/user_lerobot/NOVA
```

### Training on Nautilus cluster
LeRobot-on-Nautilus code lives under `(nautilus/training/)`. Each training container: creates a Conda env, installs FFmpeg and the right `lerobot[...]` extras, converts the dataset from v2.1 to v3.0 when needed, then runs `lerobot-train` on CUDA with Weights & Biases enabled and `policy.push_to_hub=false` (override or extend with `--train_extra`).

**Weights & Biases:** The Pod and Job templates inject `WANDB_API_KEY` from a Kubernetes secret named `wandb-secret` (key `api-key`). Create it once in your namespace so runs can log to your W&B workspace—for example:
```
kubectl create secret generic wandb-secret-nikola --from-literal=api-key=<YOUR_WANDB_API_KEY>
```
Without that secret, runs will not show up under your W&B account, or training may fail if not key is found.

| Option                  | Short | Description                                                                                                                                                                      |
| ----------------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--dataset`             | `-d`  | Hugging Face dataset repo id (required unless `--algo DUMMY`).                                                                                                                   |
| `--algo`                | `-a`  | `act` , `groot`  — supported. `pi05` — experimental in the launcher only; not supported yet. `DUMMY` — long-running sleep pod for cluster tests; ignores `--jobs`. |
| `--repeat`              | `-nr` | Number of runs, each with a random seed (default `1`).                                                                                                                           |
| `--jobs`                | `-j`  | Use a Kubernetes **Job** instead of a **Pod** (template `db-lerobot-job.yaml`).                                                                                                  |
| `--yaml_file`           | `-y`  | Custom path to Pod or Job YAML template (defaults: templates next to the launcher, `db-lerobot-pod.yaml` / `db-lerobot-job.yaml` under `nautilus/training/`).                    |
| `--namespace_pod_limit` | `-nl` | **Jobs only:** max active pods counted in the namespace; extra jobs are created **suspended** and unsuspended when capacity appears (`0` = no queuing).                          |
| `--max_concurrent`      | `-mc` | **Jobs with queuing:** cap how many of *our* jobs run at once (`0` = only the namespace limit applies).                                                                          |
| `--dry_run`             |       | Print the generated container script and exit (no `kubectl`).                                                                                                                    |
| `--state_only_act`      |       | For ACT on proprio-only data, adds `--rename_map='{"observation.state":"observation.environment_state"}'` to training.                                                           |
| `--train_extra`         |       | Single string of extra arguments appended to `lerobot-train` (quote carefully in your shell).                                                                                    |
| `--save_models`         |       | Persist training outputs: creates a timestamped `--output_dir` under the PVC (default base `--models_root`).                                                                     |
| `--models_root`         |       | Base directory on the pod for saved runs when `--save_models` is set (default `/pers_vol/dwait/saved_models/lerobot`).                                                           |


**Queuing:** If you use `--jobs` and set `--namespace_pod_limit` to a positive value, the launcher labels jobs with a queue group, starts as many as fit, and keeps unsuspending the rest as pods finish. If you interrupt that process (e.g. Ctrl+C), re-attach with `(nautilus/training/queue_watcher.py)` using the printed `--label` and the same `-nl` (and concurrency) you used at launch.

#### Start training: Ordered According to Size of the Model
- ACT
```
lerobot-train --dataset.repo_id=pollen-robotics/pick_and_place_bottle --policy.type=act --job_name=reachy2_lerobot_act --wandb.enable=false --policy.device=cuda --policy.push_to_hub=false"
```

-SmolVla
```
python nautilus/training/launch_nautilus_pods.py \
  -a smolvla \
  -d erl-hub/reachy-pick-and-place-images \
  -j \
  --train_extra "--batch_size=64 --steps=20000 --save_freq=20000 --log_freq=500 --rename_map='{"observation.images.torso_rgb":"observation.images.camera1"}' --policy.empty_cameras=2" \
  --save_models \
  --hf_model_repo erl-hub/reachy-smolvla \
  --upload_to_hub \
  -x basic-run 
```

- GR00T
```
python nautilus/training/launch_nautilus_pods.py \
  -a groot \
  -d erl-hub/reachy-pick-and-place-images \
  -j \
  --train_extra "--batch_size=64 --steps=30000 --save_freq=30000 --log_freq=500 --policy.base_model_path=nvidia/GR00T-N1.5-3B --policy.pretrained_path=/nikola_vol/saved_models/reachy/2026-04-23_21-18-51-groot-step-01-erl-hub-reachy-pick-and-place-images_s485/checkpoints/000001/pretrained_model --policy.tune_diffusion_model=true --peft.method_type=LORA --peft.r=64 --peft.target_modules='[to_q,to_k,to_v,to_out.0,proj_out_1,proj_out_2]'" \
  --save_models \
  --hf_model_repo erl-hub/reachy-groot \
  --upload_to_hub \
  -x lora-ckpt
```

```
python nautilus/training/launch_nautilus_pods.py \
  -a groot \
  -d erl-hub/reachy-pick-and-place-images \
  -j \
  --train_extra "--batch_size=64 --steps=30000 --save_freq=30000 --log_freq=500 --policy.pretrained_path=nvidia/GR00T-N1.7-3B --policy.tune_diffusion_model=true --policy.use_peft=false --peft.method_type=lora --peft.r=64 --peft.target_modules='[to_q,to_k,to_v,to_out.0,proj_out_1,proj_out_2]'" \
  --save_models \
  --hf_model_repo erl-hub/reachy-groot \
  --upload_to_hub \
  -x lora-ft
```

```
python nautilus/training/launch_nautilus_pods.py \
  -a groot \
  -d erl-hub/reachy-pick-and-place-images \
  -j \
  --train_extra "--batch_size=64 --steps=30000 --save_freq=30000 --log_freq=500 --policy.tune_diffusion_model=false" \
  --save_models \python nautilus/training/launch_nautilus_pods.py \
  -a groot \
  -d erl-hub/reachy-pick-and-place-images \
  -j \
  --train_extra "--batch_size=64 --steps=30000 --save_freq=30000 --log_freq=500 --policy.tune_diffusion_model=false" \
  --save_models \
  --hf_model_repo erl-hub/reachy-groot \
  --upload_to_hub \
  -x no-diff-ft
  --hf_model_repo erl-hub/reachy-groot \
  --upload_to_hub \
  -x no-diff-ft
```

- pi05
```
python nautilus/training/launch_nautilus_pods.py `
  -a pi05 `
  -d erl-hub/reachy-pick-and-place-images `
  -j `
  --train_extra "--batch_size=32 --steps=3000 --save_freq=3000 --log_freq=500 --policy.pretrained_path=lerobot/pi05_base --policy.compile_model=true --policy.gradient_checkpointing=true --policy.freeze_vision_encoder=true --policy.train_expert_only=false --policy.dtype=bfloat16" `
  --save_models `
  --hf_model_repo erl-hub/reachy-pi05 `
  --upload_to_hub `
  -x basic-run 
```
```
python nautilus/training/launch_nautilus_pods.py \
  -a pi05 \
  -d erl-hub/reachy-pick-and-place-images \
  -j \
  --train_extra "--batch_size=64 --steps=30000 --save_freq=30000 --log_freq=500 --policy.pretrained_path=lerobot/pi05_base --policy.use_peft=false --peft.method_type=lora --peft.r=64 --peft.target_modules='[q_proj,v_proj,k_proj,o_proj]' --policy.gradient_checkpointing=true --policy.dtype=bfloat16" \
  --save_models \
  --hf_model_repo erl-hub/reachy-pi05 \
  --upload_to_hub \
  -x lora-ft
```

- Dry-run the generated container script (no cluster submit):
```
python nautilus/training/launch_nautilus_pods.py --dry_run -a act -d pollen-robotics/pick_and_place_bottle
```

- Submit three seeded Jobs with a namespace cap of 200 pods (queued jobs unsuspend as capacity frees):
```
python nautilus/training/launch_nautilus_pods.py -j -nl 200 -nr 3 -a act -d pollen-robotics/pick_and_place_bottle
```

### Policy registry

Use this table to track policies you train or deploy. Extend rows as you add methods; replace placeholders with your checkpoints, parameter counts, and run notes.

| Policy | Method | Size | Run | Hardware | Performance (10/10) |
| ------ | ------ | ---- | --- | -------- | ------------------- |
| `act` | Action Chunking Transformer (`--policy.type=act`) | 250MB | 2026-04-14_01-02-26-erl-hub-reachy-pick-and-place-images_s7908 | CPU; Intel Core 9 (RTX GeForce 2080+) | 5 |
| `SmolVla` | VLA (`--policy.type=SmolVla`) | 1GB | 2026-04-25_05-03-29-smolvla-basic-run-erl-hub-reachy-pick-and-place-images_s2226 | NVIDIA-L40 CPU 3.8-4.0 | 6.5 |
| `GR00T` | VLA (`--policy.type=groot`) | (71.1MB)5.5GB | 2026-04-25_12-31-24-groot-lora-ckpt-erl-hub-reachy-pick-and-place-images_s2889 | NVIDIA-L40 CPU 3.8-4.0 | 2 |
| `GR00T` | VLA (`--policy.type=groot`) | (71.1MB)5.5GB | 2026-04-24_00-04-05-groot-lora-ckpt-erl-hub-reachy-pick-and-place-images_s9847 | NVIDIA-L40 CPU 1.8-2.0 | 2 |
| `GR00T` | VLA (`--policy.type=groot`) | 5.5GB | 2026-04-24_23-28-29-groot-no-diff-ft-erl-hub-reachy-pick-and-place-images_s6518 | NVIDIA-L40 CPU 3.8-4.0 30-32RAM | 2 |
| `GR00T` | VLA (`--policy.type=groot`) | 5.5GB | 2026-04-25_01-13-06-groot-diff-ft-erl-hub-reachy-pick-and-place-images_s9486 | NVIDIA-L40 CPU 3.8-4.0 30-32RAM | 3 |
| `pi05` | VLA (`--policy.type=pi05`) | (73.5MB)14.5GB | 2026-04-28_21-22-09-pi05-lora-ft-erl-hub-reachy-pick-and-place-images_s9920 | NVIDIA-L40 CPU 5.8-6.0 70-72RAM  | - |
| `pi05` | VLA (`--policy.type=pi05`) | (73.5MB)14.5GB | 2026-04-28_22-41-14-pi05-lora-ft-erl-hub-reachy-pick-and-place-images_s3055 | NVIDIA-L40S CPU 5.8-6.0 70-72RAM | - |
| `pi05` | VLA (`--policy.type=pi05`) | (73.5MB)14.5GB | 2026-04-28_22-47-15-pi05-lora-ft-erl-hub-reachy-pick-and-place-images_s536  | NVIDIA-L40S CPU 5.8-6.0 70-72RAM | - |
| `pi05` | VLA (`--policy.type=pi05`) | (73.5MB)14.5GB | 2026-04-28_23-30-24-pi05-lora-ft-erl-hub-reachy-pick-and-place-images_s9874 | NVIDIA-L40S CPU 5.8-6.0 70-72RAM | - |


#### Deploying policies on reachy2
Pass the repo for trained policy best checkpoint as `policy.path`. Videos and trajectories from rollouts will be saved at `dataset.repo_id`

```
lerobot-record \
  --robot.type=reachy2 \
  --robot.ip_address="192.168.137.100" \
  --robot.id="r2-0008" \
  --robot.use_external_commands=false \
  --robot.with_mobile_base=false \
  --robot.with_l_arm=true \
  --robot.with_r_arm=true \
  --robot.with_neck=true \
  --robot.with_antennas=false \
  --robot.with_left_teleop_camera=false \
  --robot.with_right_teleop_camera=false \
  --robot.with_torso_camera=true \
  --robot.camera_width=640 \
  --robot.camera_height=480 \
  --policy.path=erl-hub/reachy-smolvla-ft-execute-pick-and-place \
  --dataset.repo_id="erl-hub/eval_reachy-pick-and-place" \
  --dataset.root="outputs/reachy_local_test" \
  --dataset.single_task="Pick up the tomato soup can and place it into the metal bowl" \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 \
  --dataset.fps=15 \
  --dataset.vcodec=h264 \
  --dataset.push_to_hub=false \
  --play_sounds=false \
  --resume=true
```






