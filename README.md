# DexRL
Integration of BerkleyAutomation's Grasp Quality Convolutional Neural Networks (GQ-CNN) into NVIDIA`s Isaac Lab. Enhancement of Grasp Proposals from the FC-GQ-CNN with a one-step MDP RL Agent.


<img src="https://skillicons.dev/icons?i=python" /> <img src="https://skillicons.dev/icons?i=pytorch" />
<img src="https://github.com/LelouchFR/skill-icons/blob/main/assets/nvidia-auto.svg" />

Click on the video below for a Demo!

<div align="center">
  <a href="https://www.youtube.com/watch?v=BdTGw4hR2kM&ab_channel=M3NDEZ">
    <img src="https://img.youtube.com/vi/BdTGw4hR2kM/0.jpg" alt="Watch the video">
  </a>
</div>

Taking advantage of Isaac Lab's functionality for a faster parallelised training of RL agents, the integration of GQCNN was scaled to work on multiple environments independently, each with its own domain randomisation.

Below is another video - click on it for a Demo showcasing multiple environments.

<div align="center">
  <a href="https://www.youtube.com/watch?v=9fAv8-oUtLc&ab_channel=M3NDEZ">
    <img src="https://img.youtube.com/vi/9fAv8-oUtLc/0.jpg" alt="Watch the video">
  </a>
</div>


## Ambition
Today's rise in the power and fidelity to reality of simulators gives place for better and better grounds to develop RL agents. Hypothetically, with a simulator that is 100% true to reality, a Reinforcement Learning agent is only limited to its architecture and to the availability of computational power. No self-supervision is involved in RL, meaning there is no labeled dataset involved - with enough training episodes, enough computational power, and a good architecture, an agent can be trained to perform optimally at a given task.


I don't have enough knowledge about RL, enough computational resources, or an ideal simulator - however I do have ambition and a will to delve into the world of RL and simulations, so I thought it was a good idea to start by building a somewhat simple scene in Isaac Lab and developing a small agent in it.


## Aim
With this in mind, attempting to take advantage of the power of a simulator, the aim of this project is to test whether CNN-based grasp predictions, such as the ones from Dex-Net's Fully Connected GQCNN, can be improved with an RL agent. Dex-Net's FC-GQCNN predictions work approximately ~95% of the time, which is extremely impressive and still state-of-the-art today, 5 years later, in terms of overall performance and inference time.


This project attempts to:
+ close that ~5% gap when exposed to the same objects
+ see whether, if exposed to new object types (smaller, more abstract...), a simple RL agent with minimal fine tuning can help the pipeline adapt and avoid failed grasps

This second aim would, in my opinion, prove very useful in the current industry - one were smart factories are becoming more and more common, and a robot arm may be exposed to different objects over different seasons, and a minimal fine tune back in a simulator, followed by a sim2real transfer, could solve the problem.


## Methodology
The development of this project was split into two very distinct stages.
1. Full integration of DexNet's GQCNN into Isaac Lab
2. Training of a one-step SAC RL agent to enhance the GQCNN's grasp proposals

Currently, the project is in the second phase. For the RL Agent, a ManagerBasedRLEnv is being developed to wrap the whole pipeline, together with stable-baseline3's Soft-Actor-Critic policy wrapper.

Challenges included in the first phase, included:
+ Learning the Isaac Lab conventionality

In multiple aspects, from developing a simple scene to learning how to fully vectorize a Finite-State-Machine process over multiple environments.

+ Handling domain randomisation in a vectorized way

Done using RigidBodyView, handling tensors in a specific way to randomize masses, positions and dynamic / static friction.

+ Setting up Dex-Net's GQCNN module to work on my RTX 4070 GPU

The GQCNN module works on tensorflow 1.15 - the original wheel is not built to be compatible with CC 8.9 (Ada Loverance GPUs).
The NVIDIA community maintains wheels for newer GPUs / CUDAs, but only from python >= 3.8 - officialy not the case of the GQCNN module and never tested, luckily it worked out.

+ Handling the discrepancy in between the environments required by the GQCNN module and the Isaac Lab simulation

Using grequests and gevent for efficient communication between environments to set up a Client-Server configuration.


## Installation
To be able to run the project, setting up both Isaac Lab and DexNet in separate environments is necessary.

First, clone this repository and the GQCNN module inside of it.
```bash
# Clone DexRL
git clone https://github.com/imendezval/DexRL.git

cd ./DexRL/DexNet
# Clone GQCNN
git clone https://github.com/BerkeleyAutomation/gqcnn.git
```

Then, set up the environments:

#### GQCNN  
Omit steps 2 and 3 if your GPU (and its Compute Capability) is compatible with the original tensorflow 1.15 (CC <= 7.5)

1. Create and activate a Python 3.8 environment.
2. Install NVIDIA TensorFlow wheel and TensorBoard:
```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-tensorflow==1.15.5+nv23.03 nvidia-tensorboard
```
3. Comment out the get_tf_dep() functions from GQCNN's setup.py
4. Set up GQCNN module:
```bash
pip install -e .
```
5. Install required packages:
```bash
pip install Flask gevent
```

#### Isaac Lab
1. Set up the basic Isaac Lab environment using conda following the [documentation](https://isaac-sim.github.io/IsaacLab/v1.0.0/source/setup/installation/binaries_installation.html).

2. Install required packages:
```bash
pip install grequests gevent
```

## Structure
Below, the structure of the repository is represented as a tree diagram:

```bash
DexRL
├───sim_runner.py
├───scene.py
├───helpers.py
├───constants.py
├───DexNet
│   ├───DexNetServer.py
│   ├───DexNetWrapper.py
│   └───gqcnn   # the original GQCNN module
├───assets
│   ├───franka
│   ├───yumi_gripper    
│   ├───meshes
│   └───small_KLT.usd
│   └───table_instanceable.usd
├───cfg
│   ├───franka_cfg.py
│   ├───yumi_gripper.py   
└───tools
    └───OBJtoUSD.py
    └───OBJtoUSD.py
```


## Contact
For any questions or feedback, please reach out to:
- **Email**: [imendezval@stud.hs-heilbronn.de](mailto:imendezval@stud.hs-heilbronn.de) or [inigomendezval@gmail.com](mailto:inigomendezval@gmail.com)
- **GitHub Profile**: [imendezval](https://github.com/imendezval)
- **LinkedIn**: [inigo-miguel-mendez-valero](https://www.linkedin.com/in/i%C3%B1igo-miguel-m%C3%A9ndez-valero-4ba3732b1/)

Feel free to open an issue on GitHub or contact us in any way if you have any queries or suggestions.