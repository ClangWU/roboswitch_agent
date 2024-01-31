## **Setup Guidelines**
1. upgrade:
```
sudo apt update
sudo apt upgrade
```
2. Install dependencies:
```
pip install torch torchvision
pip install panda-gym 
pip install gin-config
pip install PyOpenGL 
pip install pygame gymnasium PyOpenGL_accelerate
pip install pandas 
```

## **Contents**
### Franka Panda:
This section contains the code for joystick control as well as training and testing the behavior cloning agent for the Franka Reach environment.

**Support Algorithms 支持算法**

```behavior cloning``` 

```diffusion policy```

**Quick User Guide 使用方法:**


Before running the cmds below, run
 ```mkdir -p tmp/bc && mkdir -p tmp/dp```
```
python3 collect_data.py
python3 train_bc.py
python3 test_bc.py
```
**Joystick User Guide:**
This project uses the IK controller to control the x, y & z position of the end effector. 
    - Right Joystick Vertical Motion -> Z axis
    - Right Joystick Horizontal Motion -> X axis
    - Left Joystick Horizontal Motion -> Y axis
    
