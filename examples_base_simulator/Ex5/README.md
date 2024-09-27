# FunMoro_control

This repository contains the basic simulation code to test control algorithm for ground mobile robots, namely omnidirectional robot and unicycle robot.

The model of the mobile robots rely on kinematic model of both omnidirectional robot and unicycle robot.

## Usage

Dependency: Numpy and Matplotlib

To run the code simply execute the following script:
1. ```base_code_omnidirectional.py``` : simulation code for omnidirectional robot
2. ```base_code_unicycle.py``` : simulation code for unicycle robot
3. ```base_code_omni_rangesensor.py``` : same as 1 but with range sensor to test for obstacle avoidance
4. ```base_code_unicycle_rangesensor.py``` : same as 2 but with range sensor to test for obstacle avoidance

The remaining code in the folder ```library``` contains the implementation for visuzalizing the robot icon (```library/visualize_mobile_robot.py```) and detecting obstacle (```library/detect_obstacle.py```)


**WARNING!**
The simulation code does not work with Spyder. It is highly suggested to run the code via Visual Studio or just via command line.


## Courses (Tampere University)
The previous version of this code was used for exercises in the following courses:
- AUT.710-2021-2022-1 Fundamentals of Mobile Robots (2nd part - later half)

and is planned to be used for the following courses:
- AUT.710-2022-2023-1 Fundamentals of Mobile Robots (2nd part - later half)


## Contributor and Maintainter

The basic simulation in this repository are developed and maintained by Made Widhi Surya Atman from Intelligent Networked Systems (IINES) Group, Faculty of Engineerings and Natural Sciences, Tampere University.
