import time
import numpy as np

import mujoco
import mujoco.viewer


# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(
    "models/waffle_pi.xml"
)

data = mujoco.MjData(model)


# ============================================================
# FIND WAFFLE PI BODY
# ============================================================

base_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base_link"
)


# ============================================================
# KEYBOARD
# ============================================================

current_key = None


def keyboard_callback(key):
    global current_key

    current_key = key

    print(f"Key: {key}")


# ============================================================
# WHEEL CONTROL
# ============================================================

def set_wheels(left, right):
    data.ctrl[0] = left
    data.ctrl[1] = right


def stop_robot():
    set_wheels(0.0, 0.0)


# ============================================================
# DRAW ONE ARROW
# ============================================================

def draw_arrow(scene, start, end, rgba):

    if scene.ngeom >= scene.maxgeom:
        return

    geom = scene.geoms[scene.ngeom]

    # Create arrow
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,

        np.array([
            0.015,
            0.015,
            0.015
        ]),

        np.asarray(
            start,
            dtype=np.float64
        ),

        np.eye(3).flatten(),

        np.asarray(
            rgba,
            dtype=np.float32
        )
    )

    # Set arrow direction
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.015,
        np.asarray(
            start,
            dtype=np.float64
        ),
        np.asarray(
            end,
            dtype=np.float64
        )
    )

    scene.ngeom += 1


# ============================================================
# DRAW BODY FRAME
# ============================================================

def draw_body_frame(viewer):

    # --------------------------------------------------------
    # Robot position in WORLD frame
    # --------------------------------------------------------

    position = data.xpos[base_id].copy()


    # --------------------------------------------------------
    # Rotation matrix
    # --------------------------------------------------------

    R = data.xmat[base_id].reshape(3, 3).copy()


    # --------------------------------------------------------
    # BODY AXES
    #
    # R[:,0] = body X axis
    # R[:,1] = body Y axis
    # R[:,2] = body Z axis
    # --------------------------------------------------------

    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]


    # --------------------------------------------------------
    # Start point of body frame
    # --------------------------------------------------------

    origin = position + np.array([
        0.0,
        0.0,
        0.14
    ])


    # --------------------------------------------------------
    # Axis length
    # --------------------------------------------------------

    axis_length = 0.35


    # --------------------------------------------------------
    # Calculate endpoints
    # --------------------------------------------------------

    x_end = origin + axis_length * x_axis

    y_end = origin + axis_length * y_axis

    z_end = origin + axis_length * z_axis


    # --------------------------------------------------------
    # Clear old arrows
    # --------------------------------------------------------

    viewer.user_scn.ngeom = 0


    # ========================================================
    # X AXIS
    # RED = FORWARD
    # ========================================================

    draw_arrow(
        viewer.user_scn,
        origin,
        x_end,
        [1.0, 0.0, 0.0, 1.0]
    )


    # ========================================================
    # Y AXIS
    # GREEN = LEFT
    # ========================================================

    draw_arrow(
        viewer.user_scn,
        origin,
        y_end,
        [0.0, 1.0, 0.0, 1.0]
    )


    # ========================================================
    # Z AXIS
    # BLUE = UP
    # ========================================================

    draw_arrow(
        viewer.user_scn,
        origin,
        z_end,
        [0.0, 0.3, 1.0, 1.0]
    )


# ============================================================
# DISPLAY ROTATION MATRIX IN TERMINAL
# ============================================================

def display_information():

    # Position
    position = data.xpos[base_id]

    # Rotation matrix
    R = data.xmat[base_id].reshape(3, 3)

    # Yaw angle
    yaw = np.arctan2(
        R[1, 0],
        R[0, 0]
    )

    yaw_deg = np.degrees(yaw)


    # Clear terminal
    print("\033[H\033[J", end="")


    print("================================================")
    print("       TURTLEBOT3 WAFFLE PI - MUJOCO")
    print("================================================")

    print()

    print("KEYBOARD")
    print("-----------------------------------------------")
    print("W = Forward")
    print("S = Backward")
    print("A = Rotate Left")
    print("D = Rotate Right")
    print("X = Stop")

    print()

    print("BODY FRAME")
    print("-----------------------------------------------")
    print("RED   = X body axis = Forward")
    print("GREEN = Y body axis = Left")
    print("BLUE  = Z body axis = Up")

    print()

    print("BODY POSITION")
    print("-----------------------------------------------")
    print(f"X = {position[0]: .4f} m")
    print(f"Y = {position[1]: .4f} m")
    print(f"Z = {position[2]: .4f} m")

    print()

    print("ROTATION MATRIX R")
    print("-----------------------------------------------")

    print(
        f"[ {R[0,0]: .4f}   "
        f"{R[0,1]: .4f}   "
        f"{R[0,2]: .4f} ]"
    )

    print(
        f"[ {R[1,0]: .4f}   "
        f"{R[1,1]: .4f}   "
        f"{R[1,2]: .4f} ]"
    )

    print(
        f"[ {R[2,0]: .4f}   "
        f"{R[2,1]: .4f}   "
        f"{R[2,2]: .4f} ]"
    )

    print()

    print(
        f"YAW = {yaw_deg:.2f} degrees"
    )

    print("================================================")


# ============================================================
# START
# ============================================================

print("Starting MuJoCo...")


with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=keyboard_callback
) as viewer:

    # ========================================================
    # CAMERA
    # ========================================================

    viewer.cam.azimuth = 90

    viewer.cam.elevation = -20

    viewer.cam.distance = 1.4

    viewer.cam.lookat[0] = 0.0
    viewer.cam.lookat[1] = 0.0
    viewer.cam.lookat[2] = 0.10


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while viewer.is_running():

        # ====================================================
        # KEYBOARD CONTROL
        # ====================================================

        if current_key == 87:

            # W = forward
            set_wheels(
                3.0,
                3.0
            )


        elif current_key == 83:

            # S = backward
            set_wheels(
                -3.0,
                -3.0
            )


        elif current_key == 65:

            # A = rotate left
            set_wheels(
                -2.0,
                2.0
            )


        elif current_key == 68:

            # D = rotate right
            set_wheels(
                2.0,
                -2.0
            )


        elif current_key == 88:

            # X = stop
            stop_robot()


        # ====================================================
        # PHYSICS
        # ====================================================

        mujoco.mj_step(
            model,
            data
        )


        # ====================================================
        # DRAW BODY FRAME
        # ====================================================

        draw_body_frame(
            viewer
        )


        # ====================================================
        # PRINT MATRIX
        # ====================================================

        display_information()


        # ====================================================
        # UPDATE VIEWER
        # ====================================================

        viewer.sync()


        # ====================================================
        # SIMULATION SPEED
        # ====================================================

        time.sleep(0.01)