cd ~/turtlebot3_mujoco

cat > README.md <<'EOF'
# Lab 1 — MuJoCo Robotics Challenges

## Overview

This repository contains the solutions for Lab 1 using the MuJoCo physics simulator.

## Challenges

### Challenge 1 — TurtleBot3 Waffle Pi

A TurtleBot3 Waffle Pi model implemented and simulated in MuJoCo.

Files:
- `challenges/challenge1/main.py`
- `challenges/challenge1/waffle_pi.xml`

### Challenge 2 — Quadrotor

A quadrotor model implemented and simulated in MuJoCo.

Files:
- `challenges/challenge2/main_quad.py`
- `challenges/challenge2/quadrotor.xml`

## Project Structure

```text
lab1_turtlebot3_mujoco/
├── challenges/
│   ├── challenge1/
│   │   ├── main.py
│   │   └── waffle_pi.xml
│   └── challenge2/
│       ├── main_quad.py
│       └── quadrotor.xml
├── .gitignore
└── README.md
