import numpy as np
import tkinter as tk
import mujoco
import glfw


# ============================================================
# QUADROTOR - CHALLENGE 2
# ============================================================
#
# CONTROLS
#
# UP       = FORWARD
# DOWN     = BACKWARD
# LEFT     = LEFT
# RIGHT    = RIGHT
#
# W        = ASCEND
# S        = DESCEND
# SPACE    = STOP / LEVEL
# ESC      = EXIT
#
# CAMERA
#
# Left mouse  = Orbit
# Right mouse = Pan
# Wheel       = Zoom
#
# FRAME
#
# X RED    = Forward / Lane
# Y GREEN  = Left / Right
# Z BLUE   = Up
#
# ============================================================


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "models/quadrotor.xml"

print("Loading quadrotor model...")

model = mujoco.MjModel.from_xml_path(
    MODEL_PATH
)

data = mujoco.MjData(model)


# ============================================================
# BASE BODY
# ============================================================

base_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base_link"
)

if base_id < 0:
    raise RuntimeError(
        "Could not find body 'base_link' in quadrotor.xml"
    )


# ============================================================
# FREE JOINT
# ============================================================

free_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "quadrotor_free"
)

if free_joint < 0:
    raise RuntimeError(
        "Could not find joint 'quadrotor_free'"
    )


free_qpos = model.jnt_qposadr[free_joint]

free_dof = model.jnt_dofadr[free_joint]


# ============================================================
# MASS
# ============================================================

MASS = float(
    model.body_mass[base_id]
)

GRAVITY = 9.81


# ============================================================
# INITIAL POSITION
# ============================================================

INITIAL_X = 0.0
INITIAL_Y = 0.0
INITIAL_Z = 1.0


# ============================================================
# TARGET ALTITUDE
# ============================================================

TARGET_Z = INITIAL_Z


# ============================================================
# ATTITUDE LIMIT
# ============================================================

MAX_TILT_DEG = 12.0

MAX_TILT = np.radians(
    MAX_TILT_DEG
)


# ============================================================
# ALTITUDE CONTROL
# ============================================================

ALTITUDE_STEP = 0.04

ALTITUDE_KP = 5.0

ALTITUDE_KD = 3.2

MAX_VERTICAL_ACCEL = 3.0


# ============================================================
# ATTITUDE CONTROL
# ============================================================

ROLL_KP = 4.5
ROLL_KD = 2.0

PITCH_KP = 4.5
PITCH_KD = 2.0

YAW_KP = 1.2
YAW_KD = 0.9


# ============================================================
# TORQUE LIMITS
# ============================================================

MAX_ROLL_TORQUE = 1.0

MAX_PITCH_TORQUE = 1.0

MAX_YAW_TORQUE = 0.30


# ============================================================
# THRUST
# ============================================================

HOVER_THRUST = (
    MASS * GRAVITY
)

MIN_TOTAL_THRUST = 0.0

MAX_TOTAL_THRUST = (
    HOVER_THRUST * 1.35
)


# ============================================================
# PHYSICAL JITTER
# ============================================================

JITTER_ENABLED = True

JITTER_FORCE = 0.025

JITTER_TORQUE = 0.008

JITTER_FREQUENCY = 1.7

STARTUP_JITTER_DURATION = 1.5

startup_time = 0.0


# ============================================================
# KEYBOARD STATE
# ============================================================

up_pressed = False
down_pressed = False
left_pressed = False
right_pressed = False

w_pressed = False
s_pressed = False

space_pressed = False


# ============================================================
# DESIRED ATTITUDE
# ============================================================

desired_roll = 0.0

desired_pitch = 0.0

desired_yaw = 0.0


# ============================================================
# STATUS
# ============================================================

current_status = "HOVER"


# ============================================================
# MOTOR DISPLAY
# ============================================================

motor_front = 0.0
motor_back = 0.0
motor_left = 0.0
motor_right = 0.0


# ============================================================
# CAMERA
# ============================================================

FOLLOW_ROBOT = True

CAMERA_DISTANCE = 2.0

CAMERA_ELEVATION = -20.0

CAMERA_AZIMUTH = 90.0

CAMERA_HEIGHT = 0.10


camera = None
option = None
scene = None
context = None


# ============================================================
# MOUSE STATE
# ============================================================

mouse_left_pressed = False

mouse_right_pressed = False

last_mouse_x = 0.0

last_mouse_y = 0.0


# ============================================================
# ROTATION MATRIX
# ============================================================

def get_rotation_matrix():

    return data.xmat[
        base_id
    ].reshape(
        3,
        3
    ).copy()


# ============================================================
# EULER ANGLES
# ============================================================

def get_euler_angles():

    R = get_rotation_matrix()

    roll = np.arctan2(
        R[2, 1],
        R[2, 2]
    )

    pitch = np.arcsin(
        np.clip(
            -R[2, 0],
            -1.0,
            1.0
        )
    )

    yaw = np.arctan2(
        R[1, 0],
        R[0, 0]
    )

    return (
        roll,
        pitch,
        yaw
    )


# ============================================================
# QUATERNION FROM EULER
# ============================================================

def quaternion_from_euler(
    roll,
    pitch,
    yaw
):

    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)

    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)

    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)

    return np.array([

        cr * cp * cy
        + sr * sp * sy,

        sr * cp * cy
        - cr * sp * sy,

        cr * sp * cy
        + sr * cp * sy,

        cr * cp * sy
        - sr * sp * cy

    ])


# ============================================================
# SET POSE
# ============================================================

def set_pose(
    x,
    y,
    z,
    roll_deg,
    pitch_deg,
    yaw_deg
):

    global TARGET_Z
    global desired_roll
    global desired_pitch
    global desired_yaw
    global startup_time

    mujoco.mj_resetData(
        model,
        data
    )

    data.qpos[
        free_qpos:
        free_qpos + 3
    ] = [
        x,
        y,
        z
    ]

    data.qpos[
        free_qpos + 3:
        free_qpos + 7
    ] = quaternion_from_euler(

        np.radians(roll_deg),
        np.radians(pitch_deg),
        np.radians(yaw_deg)

    )

    data.qvel[:] = 0.0

    data.ctrl[:] = 0.0

    data.xfrc_applied[:] = 0.0

    TARGET_Z = z

    desired_roll = np.radians(
        roll_deg
    )

    desired_pitch = np.radians(
        pitch_deg
    )

    desired_yaw = np.radians(
        yaw_deg
    )

    startup_time = data.time

    mujoco.mj_forward(
        model,
        data
    )


# ============================================================
# RESET
# ============================================================

def reset_origin():

    set_pose(
        0.0,
        0.0,
        INITIAL_Z,
        0.0,
        0.0,
        0.0
    )

    update_entries(
        0.0,
        0.0,
        INITIAL_Z,
        0.0,
        0.0,
        0.0
    )


# ============================================================
# KEYBOARD CALLBACK
# ============================================================

def keyboard_callback(
    window,
    key,
    scancode,
    action,
    mods
):

    global up_pressed
    global down_pressed
    global left_pressed
    global right_pressed

    global w_pressed
    global s_pressed
    global space_pressed

    # ========================================================
    # ESC
    # ========================================================

    if key == glfw.KEY_ESCAPE:

        if action == glfw.PRESS:

            glfw.set_window_should_close(
                window,
                True
            )

        return

    # ========================================================
    # PRESS
    # ========================================================

    if action == glfw.PRESS:

        if key == glfw.KEY_UP:

            up_pressed = True

        elif key == glfw.KEY_DOWN:

            down_pressed = True

        elif key == glfw.KEY_LEFT:

            left_pressed = True

        elif key == glfw.KEY_RIGHT:

            right_pressed = True

        elif key == glfw.KEY_W:

            w_pressed = True

        elif key == glfw.KEY_S:

            s_pressed = True

        elif key == glfw.KEY_SPACE:

            space_pressed = True

    # ========================================================
    # RELEASE
    # ========================================================

    elif action == glfw.RELEASE:

        if key == glfw.KEY_UP:

            up_pressed = False

        elif key == glfw.KEY_DOWN:

            down_pressed = False

        elif key == glfw.KEY_LEFT:

            left_pressed = False

        elif key == glfw.KEY_RIGHT:

            right_pressed = False

        elif key == glfw.KEY_W:

            w_pressed = False

        elif key == glfw.KEY_S:

            s_pressed = False

        elif key == glfw.KEY_SPACE:

            space_pressed = False


# ============================================================
# COMMAND UPDATE
#
# IMPORTANT:
#
# Based on the behavior you described, your previous
# controller's physical axes were rotated:
#
# OLD LEFT  -> FORWARD
# OLD RIGHT -> BACKWARD
# OLD UP    -> LEFT
# OLD DOWN  -> RIGHT
#
# Therefore:
#
# NEW UP    -> uses old LEFT  -> FORWARD
# NEW DOWN  -> uses old RIGHT -> BACKWARD
# NEW LEFT  -> uses old UP    -> LEFT
# NEW RIGHT -> uses old DOWN  -> RIGHT
#
# ============================================================

def update_command():

    global desired_roll
    global desired_pitch
    global current_status

    # ========================================================
    # SPACE = STOP / LEVEL
    # ========================================================

    if space_pressed:

        desired_roll = 0.0

        desired_pitch = 0.0

        current_status = "STOP / LEVEL"

        return


    # ========================================================
    # UP / DOWN = FORWARD / BACKWARD
    #
    # These use the old LEFT / RIGHT physical response.
    # ========================================================

    if up_pressed and not down_pressed:

        # Old LEFT command
        desired_roll = -MAX_TILT

    elif down_pressed and not up_pressed:

        # Old RIGHT command
        desired_roll = MAX_TILT

    else:

        desired_roll *= 0.92


    # ========================================================
    # LEFT / RIGHT = LEFT / RIGHT
    #
    # These use the old UP / DOWN physical response.
    # ========================================================

    if left_pressed and not right_pressed:

        # Old UP command
        desired_pitch = MAX_TILT

    elif right_pressed and not left_pressed:

        # Old DOWN command
        desired_pitch = -MAX_TILT

    else:

        desired_pitch *= 0.92


    # ========================================================
    # LIMIT
    # ========================================================

    desired_roll = np.clip(
        desired_roll,
        -MAX_TILT,
        MAX_TILT
    )

    desired_pitch = np.clip(
        desired_pitch,
        -MAX_TILT,
        MAX_TILT
    )


    # ========================================================
    # STATUS
    # ========================================================

    if up_pressed and left_pressed:

        current_status = (
            "FORWARD + LEFT"
        )

    elif up_pressed and right_pressed:

        current_status = (
            "FORWARD + RIGHT"
        )

    elif down_pressed and left_pressed:

        current_status = (
            "BACKWARD + LEFT"
        )

    elif down_pressed and right_pressed:

        current_status = (
            "BACKWARD + RIGHT"
        )

    elif up_pressed:

        current_status = (
            "FORWARD"
        )

    elif down_pressed:

        current_status = (
            "BACKWARD"
        )

    elif left_pressed:

        current_status = (
            "LEFT"
        )

    elif right_pressed:

        current_status = (
            "RIGHT"
        )

    elif w_pressed:

        current_status = (
            "ASCEND"
        )

    elif s_pressed:

        current_status = (
            "DESCEND"
        )

    else:

        current_status = (
            "HOVER"
        )


# ============================================================
# ALTITUDE CONTROL
# ============================================================

def calculate_total_thrust():

    global TARGET_Z

    if space_pressed:

        return 0.0

    z = data.xpos[
        base_id,
        2
    ]

    vz = data.qvel[
        free_dof + 2
    ]


    # ========================================================
    # W = ASCEND
    # ========================================================

    if w_pressed and not s_pressed:

        TARGET_Z += (
            ALTITUDE_STEP
            * model.opt.timestep
            * 50.0
        )


    # ========================================================
    # S = DESCEND
    # ========================================================

    elif s_pressed and not w_pressed:

        TARGET_Z -= (
            ALTITUDE_STEP
            * model.opt.timestep
            * 50.0
        )


    TARGET_Z = np.clip(
        TARGET_Z,
        0.35,
        5.0
    )


    error = (
        TARGET_Z - z
    )


    acceleration = (
        ALTITUDE_KP * error
        -
        ALTITUDE_KD * vz
    )


    acceleration = np.clip(
        acceleration,
        -MAX_VERTICAL_ACCEL,
        MAX_VERTICAL_ACCEL
    )


    thrust = (
        MASS
        * (
            GRAVITY
            + acceleration
        )
    )


    return np.clip(
        thrust,
        MIN_TOTAL_THRUST,
        MAX_TOTAL_THRUST
    )


# ============================================================
# ATTITUDE TORQUE
# ============================================================

def calculate_attitude_torque():

    roll, pitch, yaw = (
        get_euler_angles()
    )

    R = get_rotation_matrix()


    angular_velocity_world = (
        data.qvel[
            free_dof + 3:
            free_dof + 6
        ]
    )


    angular_velocity_body = (
        R.T
        @ angular_velocity_world
    )


    roll_rate = (
        angular_velocity_body[0]
    )

    pitch_rate = (
        angular_velocity_body[1]
    )

    yaw_rate = (
        angular_velocity_body[2]
    )


    # ========================================================
    # ROLL
    # ========================================================

    roll_torque = (
        ROLL_KP
        * (
            desired_roll
            - roll
        )
        -
        ROLL_KD
        * roll_rate
    )


    # ========================================================
    # PITCH
    # ========================================================

    pitch_torque = (
        PITCH_KP
        * (
            desired_pitch
            - pitch
        )
        -
        PITCH_KD
        * pitch_rate
    )


    # ========================================================
    # YAW
    # ========================================================

    yaw_error = (
        desired_yaw
        - yaw
    )


    yaw_error = np.arctan2(
        np.sin(yaw_error),
        np.cos(yaw_error)
    )


    yaw_torque = (
        YAW_KP
        * yaw_error
        -
        YAW_KD
        * yaw_rate
    )


    roll_torque = np.clip(
        roll_torque,
        -MAX_ROLL_TORQUE,
        MAX_ROLL_TORQUE
    )


    pitch_torque = np.clip(
        pitch_torque,
        -MAX_PITCH_TORQUE,
        MAX_PITCH_TORQUE
    )


    yaw_torque = np.clip(
        yaw_torque,
        -MAX_YAW_TORQUE,
        MAX_YAW_TORQUE
    )


    return (
        roll_torque,
        pitch_torque,
        yaw_torque
    )


# ============================================================
# APPLY QUADROTOR FORCE / TORQUE
# ============================================================

def apply_quadrotor_forces():

    global motor_front
    global motor_back
    global motor_left
    global motor_right


    data.xfrc_applied[:] = 0.0


    if space_pressed:

        motor_front = 0.0
        motor_back = 0.0
        motor_left = 0.0
        motor_right = 0.0

        return


    total_thrust = (
        calculate_total_thrust()
    )


    roll_torque, pitch_torque, yaw_torque = (
        calculate_attitude_torque()
    )


    R = get_rotation_matrix()


    # ========================================================
    # BODY Z -> WORLD
    # ========================================================

    thrust_direction = R[:, 2]


    force_world = (
        thrust_direction
        * total_thrust
    )


    # ========================================================
    # FORCE
    # ========================================================

    data.xfrc_applied[
        base_id,
        0:3
    ] = force_world


    # ========================================================
    # TORQUES
    # ========================================================

    data.xfrc_applied[
        base_id,
        3
    ] = roll_torque

    data.xfrc_applied[
        base_id,
        4
    ] = pitch_torque

    data.xfrc_applied[
        base_id,
        5
    ] = yaw_torque


    # ========================================================
    # MOTOR DISPLAY
    # ========================================================

    motor_front = max(
        0.0,
        total_thrust / 4.0
        - pitch_torque * 0.15
        + yaw_torque * 0.05
    )


    motor_back = max(
        0.0,
        total_thrust / 4.0
        + pitch_torque * 0.15
        - yaw_torque * 0.05
    )


    motor_left = max(
        0.0,
        total_thrust / 4.0
        - roll_torque * 0.15
        - yaw_torque * 0.05
    )


    motor_right = max(
        0.0,
        total_thrust / 4.0
        + roll_torque * 0.15
        + yaw_torque * 0.05
    )


# ============================================================
# PHYSICAL JITTER
# ============================================================

def apply_physical_jitter():

    if not JITTER_ENABLED:

        return


    elapsed = (
        data.time
        - startup_time
    )


    if elapsed < STARTUP_JITTER_DURATION:

        strength = 1.0

    else:

        strength = 0.30


    t = data.time


    fx = (
        JITTER_FORCE
        * strength
        * np.sin(
            2.0
            * np.pi
            * JITTER_FREQUENCY
            * t
        )
    )


    fy = (
        JITTER_FORCE
        * strength
        * np.sin(
            2.0
            * np.pi
            * JITTER_FREQUENCY
            * 0.73
            * t
            + 1.2
        )
    )


    fz = (
        JITTER_FORCE
        * 0.35
        * strength
        * np.sin(
            2.0
            * np.pi
            * JITTER_FREQUENCY
            * 1.31
            * t
        )
    )


    tx = (
        JITTER_TORQUE
        * strength
        * np.sin(
            2.0
            * np.pi
            * JITTER_FREQUENCY
            * 0.91
            * t
        )
    )


    ty = (
        JITTER_TORQUE
        * strength
        * np.sin(
            2.0
            * np.pi
            * JITTER_FREQUENCY
            * 1.13
            * t
            + 0.7
        )
    )


    tz = (
        JITTER_TORQUE
        * 0.5
        * strength
        * np.sin(
            2.0
            * np.pi
            * JITTER_FREQUENCY
            * 0.67
            * t
        )
    )


    data.xfrc_applied[
        base_id,
        0
    ] += fx

    data.xfrc_applied[
        base_id,
        1
    ] += fy

    data.xfrc_applied[
        base_id,
        2
    ] += fz

    data.xfrc_applied[
        base_id,
        3
    ] += tx

    data.xfrc_applied[
        base_id,
        4
    ] += ty

    data.xfrc_applied[
        base_id,
        5
    ] += tz


# ============================================================
# DRAW ARROW
# ============================================================

def draw_arrow(
    start,
    end,
    color,
    width=0.025
):

    if scene.ngeom >= scene.maxgeom:

        return


    geom = scene.geoms[
        scene.ngeom
    ]


    mujoco.mjv_initGeom(

        geom,

        mujoco.mjtGeom.mjGEOM_ARROW,

        np.array([
            width,
            width,
            width
        ]),

        np.asarray(
            start,
            dtype=np.float64
        ),

        np.eye(3).flatten(),

        np.asarray(
            color,
            dtype=np.float32
        )

    )


    mujoco.mjv_connector(

        geom,

        mujoco.mjtGeom.mjGEOM_ARROW,

        width,

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
# DRAW WORLD FRAME
# ============================================================

def draw_world_frame():

    origin = np.array([
        0.0,
        0.0,
        0.02
    ])

    length = 0.8


    # ========================================================
    # WORLD X = RED
    # ========================================================

    draw_arrow(

        origin,

        origin + np.array([
            length,
            0.0,
            0.0
        ]),

        [1.0, 0.0, 0.0, 1.0]

    )


    # ========================================================
    # WORLD Y = GREEN
    # ========================================================

    draw_arrow(

        origin,

        origin + np.array([
            0.0,
            length,
            0.0
        ]),

        [0.0, 1.0, 0.0, 1.0]

    )


    # ========================================================
    # WORLD Z = BLUE
    # ========================================================

    draw_arrow(

        origin,

        origin + np.array([
            0.0,
            0.0,
            length
        ]),

        [0.0, 0.3, 1.0, 1.0]

    )


# ============================================================
# DRAW BODY FRAME
# ============================================================

def draw_body_frame():

    position = data.xpos[
        base_id
    ].copy()


    R = get_rotation_matrix()


    origin = (
        position
        + np.array([
            0.0,
            0.0,
            0.20
        ])
    )


    length = 0.55


    # ========================================================
    # BODY X = RED
    # ========================================================

    draw_arrow(

        origin,

        origin
        + length * R[:, 0],

        [1.0, 0.0, 0.0, 1.0],

        0.03

    )


    # ========================================================
    # BODY Y = GREEN
    # ========================================================

    draw_arrow(

        origin,

        origin
        + length * R[:, 1],

        [0.0, 1.0, 0.0, 1.0],

        0.03

    )


    # ========================================================
    # BODY Z = BLUE
    # ========================================================

    draw_arrow(

        origin,

        origin
        + length * R[:, 2],

        [0.0, 0.3, 1.0, 1.0],

        0.03

    )


# ============================================================
# MOUSE BUTTON CALLBACK
# ============================================================

def mouse_button_callback(
    window,
    button,
    action,
    mods
):

    global mouse_left_pressed
    global mouse_right_pressed
    global last_mouse_x
    global last_mouse_y


    if action == glfw.PRESS:

        last_mouse_x, last_mouse_y = (
            glfw.get_cursor_pos(
                window
            )
        )


        if button == glfw.MOUSE_BUTTON_LEFT:

            mouse_left_pressed = True


        elif button == glfw.MOUSE_BUTTON_RIGHT:

            mouse_right_pressed = True


    elif action == glfw.RELEASE:

        if button == glfw.MOUSE_BUTTON_LEFT:

            mouse_left_pressed = False


        elif button == glfw.MOUSE_BUTTON_RIGHT:

            mouse_right_pressed = False


# ============================================================
# MOUSE MOVE
# ============================================================

def cursor_position_callback(
    window,
    xpos,
    ypos
):

    global last_mouse_x
    global last_mouse_y

    global CAMERA_AZIMUTH
    global CAMERA_ELEVATION

    global FOLLOW_ROBOT


    dx = (
        xpos
        - last_mouse_x
    )

    dy = (
        ypos
        - last_mouse_y
    )


    last_mouse_x = xpos

    last_mouse_y = ypos


    # ========================================================
    # LEFT DRAG = ORBIT
    # ========================================================

    if mouse_left_pressed:

        CAMERA_AZIMUTH += (
            dx * 0.4
        )

        CAMERA_ELEVATION += (
            dy * 0.25
        )


        CAMERA_ELEVATION = np.clip(
            CAMERA_ELEVATION,
            -89.0,
            89.0
        )


        camera.azimuth = (
            CAMERA_AZIMUTH
        )

        camera.elevation = (
            CAMERA_ELEVATION
        )


    # ========================================================
    # RIGHT DRAG = PAN
    # ========================================================

    elif mouse_right_pressed:

        FOLLOW_ROBOT = False

        camera.lookat[0] -= (
            dx * 0.005
        )

        camera.lookat[1] += (
            dy * 0.005
        )


# ============================================================
# MOUSE SCROLL
# ============================================================

def scroll_callback(
    window,
    xoffset,
    yoffset
):

    global CAMERA_DISTANCE


    CAMERA_DISTANCE -= (
        yoffset
        * 0.15
    )


    CAMERA_DISTANCE = np.clip(
        CAMERA_DISTANCE,
        0.5,
        20.0
    )


    camera.distance = (
        CAMERA_DISTANCE
    )


# ============================================================
# TKINTER WINDOW
# ============================================================

info_window = tk.Tk()

info_window.title(
    "Quadrotor - Live Rotation Matrix"
)

info_window.geometry(
    "720x900"
)


# ============================================================
# TITLE
# ============================================================

tk.Label(

    info_window,

    text="QUADROTOR - CHALLENGE 2",

    font=(
        "Arial",
        18,
        "bold"
    )

).pack(
    pady=8
)


# ============================================================
# SPAWN FRAME
# ============================================================

spawn_frame = tk.LabelFrame(

    info_window,

    text="RESPAWN / INITIAL CONDITION"

)

spawn_frame.pack(

    padx=10,
    pady=8,
    fill=tk.X

)


# ============================================================
# ENTRIES
# ============================================================

field_entries = {}


field_list = [

    ("X", "0.000"),
    ("Y", "0.000"),
    ("Z", "1.000"),
    ("Roll", "0.000"),
    ("Pitch", "0.000"),
    ("Yaw", "0.000")

]


for i, (
    name,
    value
) in enumerate(field_list):

    row = i // 3

    column = (
        i % 3
    ) * 2


    tk.Label(

        spawn_frame,

        text=name

    ).grid(

        row=row,

        column=column,

        padx=5,

        pady=5

    )


    entry = tk.Entry(

        spawn_frame,

        width=10

    )


    entry.insert(
        0,
        value
    )


    entry.grid(

        row=row,

        column=column + 1,

        padx=5,

        pady=5

    )


    field_entries[
        name
    ] = entry


# ============================================================
# UPDATE ENTRIES
# ============================================================

def update_entries(
    x,
    y,
    z,
    roll,
    pitch,
    yaw
):

    values = {

        "X": x,
        "Y": y,
        "Z": z,
        "Roll": roll,
        "Pitch": pitch,
        "Yaw": yaw

    }


    for name, value in values.items():

        field_entries[
            name
        ].delete(
            0,
            tk.END
        )


        field_entries[
            name
        ].insert(
            0,
            f"{value:.3f}"
        )


# ============================================================
# APPLY POSE
# ============================================================

def apply_pose():

    try:

        x = float(
            field_entries["X"].get()
        )

        y = float(
            field_entries["Y"].get()
        )

        z = float(
            field_entries["Z"].get()
        )

        roll = float(
            field_entries["Roll"].get()
        )

        pitch = float(
            field_entries["Pitch"].get()
        )

        yaw = float(
            field_entries["Yaw"].get()
        )


        set_pose(

            x,
            y,
            z,
            roll,
            pitch,
            yaw

        )


    except ValueError:

        print(
            "ERROR: Enter valid numbers."
        )


# ============================================================
# RANDOM POSE
# ============================================================

def random_pose():

    x = np.random.uniform(
        -2.0,
        2.0
    )

    y = np.random.uniform(
        -2.0,
        2.0
    )

    z = np.random.uniform(
        0.8,
        2.0
    )

    roll = np.random.uniform(
        -5.0,
        5.0
    )

    pitch = np.random.uniform(
        -5.0,
        5.0
    )

    yaw = np.random.uniform(
        -180.0,
        180.0
    )


    set_pose(

        x,
        y,
        z,
        roll,
        pitch,
        yaw

    )


    update_entries(

        x,
        y,
        z,
        roll,
        pitch,
        yaw

    )


# ============================================================
# BUTTONS
# ============================================================

button_frame = tk.Frame(
    spawn_frame
)


button_frame.grid(

    row=2,
    column=0,
    columnspan=6,
    pady=8

)


tk.Button(

    button_frame,

    text="APPLY POSE",

    command=apply_pose,

    width=15

).pack(

    side=tk.LEFT,
    padx=5

)


tk.Button(

    button_frame,

    text="RESET ORIGIN",

    command=reset_origin,

    width=15

).pack(

    side=tk.LEFT,
    padx=5

)


tk.Button(

    button_frame,

    text="RANDOM POSE",

    command=random_pose,

    width=15

).pack(

    side=tk.LEFT,
    padx=5

)


# ============================================================
# CAMERA CONTROL FRAME
# ============================================================

camera_frame = tk.LabelFrame(

    info_window,

    text="CAMERA"

)

camera_frame.pack(

    padx=10,
    pady=5,
    fill=tk.X

)


follow_var = tk.BooleanVar(
    value=True
)


def toggle_follow():

    global FOLLOW_ROBOT

    FOLLOW_ROBOT = (
        follow_var.get()
    )


tk.Checkbutton(

    camera_frame,

    text="FOLLOW ROBOT",

    variable=follow_var,

    command=toggle_follow

).pack(

    side=tk.LEFT,
    padx=10

)


tk.Label(

    camera_frame,

    text=(
        "Left drag = Orbit    "
        "Right drag = Pan    "
        "Wheel = Zoom"
    )

).pack(

    side=tk.LEFT,
    padx=5

)


# ============================================================
# MATRIX FRAME
# ============================================================

matrix_frame = tk.LabelFrame(

    info_window,

    text="LIVE ROBOT POSE / ROTATION MATRIX"

)

matrix_frame.pack(

    padx=10,
    pady=8,
    fill=tk.BOTH,
    expand=True

)


matrix_text = tk.Text(

    matrix_frame,

    font=(
        "Courier New",
        11
    ),

    bg="black",

    fg="white",

    insertbackground="white"

)


matrix_text.pack(

    padx=8,
    pady=8,
    fill=tk.BOTH,
    expand=True

)


# ============================================================
# MATRIX UPDATE
# ============================================================

def update_matrix_window():

    position = data.xpos[
        base_id
    ].copy()


    R_WB = get_rotation_matrix()

    R_BW = R_WB.T


    roll, pitch, yaw = (
        get_euler_angles()
    )


    elapsed = (
        data.time
        - startup_time
    )


    if elapsed < STARTUP_JITTER_DURATION:

        jitter_status = (
            "STARTUP PHYSICAL SETTLING"
        )

    else:

        jitter_status = (
            "NORMAL PHYSICAL MICRO-JITTER"
        )


    text = f"""

========================================================
                 QUADROTOR CHALLENGE 2
========================================================

POSITION
--------------------------------------------------------
X       = {position[0]: .4f} m
Y       = {position[1]: .4f} m
Z       = {position[2]: .4f} m


ORIENTATION
--------------------------------------------------------
ROLL    = {np.degrees(roll): .3f} deg
PITCH   = {np.degrees(pitch): .3f} deg
YAW     = {np.degrees(yaw): .3f} deg


R_WB : BODY -> WORLD
--------------------------------------------------------
[ {R_WB[0,0]: .4f}  {R_WB[0,1]: .4f}  {R_WB[0,2]: .4f} ]
[ {R_WB[1,0]: .4f}  {R_WB[1,1]: .4f}  {R_WB[1,2]: .4f} ]
[ {R_WB[2,0]: .4f}  {R_WB[2,1]: .4f}  {R_WB[2,2]: .4f} ]


R_BW : WORLD -> BODY
--------------------------------------------------------
[ {R_BW[0,0]: .4f}  {R_BW[0,1]: .4f}  {R_BW[0,2]: .4f} ]
[ {R_BW[1,0]: .4f}  {R_BW[1,1]: .4f}  {R_BW[1,2]: .4f} ]
[ {R_BW[2,0]: .4f}  {R_BW[2,1]: .4f}  {R_BW[2,2]: .4f} ]


BODY AXES IN WORLD FRAME
--------------------------------------------------------
BODY X = (
 {R_WB[0,0]: .4f},
 {R_WB[1,0]: .4f},
 {R_WB[2,0]: .4f}
)

BODY Y = (
 {R_WB[0,1]: .4f},
 {R_WB[1,1]: .4f},
 {R_WB[2,1]: .4f}
)

BODY Z = (
 {R_WB[0,2]: .4f},
 {R_WB[1,2]: .4f},
 {R_WB[2,2]: .4f}
)


CONTROL
--------------------------------------------------------
STATUS = {current_status}

TARGET Z = {TARGET_Z:.3f} m

DESIRED ROLL  = {np.degrees(desired_roll): .2f} deg

DESIRED PITCH = {np.degrees(desired_pitch): .2f} deg


MOTOR THRUST
--------------------------------------------------------
FRONT = {motor_front:.4f} N
BACK  = {motor_back:.4f} N
LEFT  = {motor_left:.4f} N
RIGHT = {motor_right:.4f} N


PHYSICAL JITTER
--------------------------------------------------------
{jitter_status}

Jitter force  = {JITTER_FORCE:.4f}

Jitter torque = {JITTER_TORQUE:.4f}


CONTROLS
--------------------------------------------------------
UP       = FORWARD
DOWN     = BACKWARD

LEFT     = LEFT
RIGHT    = RIGHT

W        = ASCEND
S        = DESCEND

SPACE    = STOP / LEVEL


CAMERA
--------------------------------------------------------
Left mouse  = Orbit
Right mouse = Pan
Wheel       = Zoom


FRAME
--------------------------------------------------------
X RED    = Forward / Lane
Y GREEN  = Left / Right
Z BLUE   = Up / Sky


SIMULATION TIME
--------------------------------------------------------
{data.time:.3f} s

========================================================

"""


    matrix_text.delete(
        "1.0",
        tk.END
    )


    matrix_text.insert(
        tk.END,
        text
    )


# ============================================================
# INITIALIZE GLFW
# ============================================================

if not glfw.init():

    raise RuntimeError(
        "Could not initialize GLFW"
    )


# ============================================================
# CREATE WINDOW
# ============================================================

window = glfw.create_window(

    1400,
    900,

    "Quadrotor - Challenge 2",

    None,
    None

)


if not window:

    glfw.terminate()

    raise RuntimeError(
        "Could not create GLFW window"
    )


# ============================================================
# GLFW CONTEXT
# ============================================================

glfw.make_context_current(
    window
)

glfw.swap_interval(
    1
)


# ============================================================
# CALLBACKS
# ============================================================

glfw.set_key_callback(
    window,
    keyboard_callback
)

glfw.set_mouse_button_callback(
    window,
    mouse_button_callback
)

glfw.set_cursor_pos_callback(
    window,
    cursor_position_callback
)

glfw.set_scroll_callback(
    window,
    scroll_callback
)


# ============================================================
# MUJOCO VISUALIZATION
# ============================================================

camera = mujoco.MjvCamera()

option = mujoco.MjvOption()

scene = mujoco.MjvScene(
    model,
    maxgeom=2000
)

context = mujoco.MjrContext(
    model,
    mujoco.mjtFontScale.mjFONTSCALE_100
)


# ============================================================
# DEFAULT CAMERA
# ============================================================

mujoco.mjv_defaultCamera(
    camera
)

mujoco.mjv_defaultOption(
    option
)


# ============================================================
# WAFFLE PI STYLE CAMERA
# ============================================================

camera.azimuth = 90.0

camera.elevation = -20.0

camera.distance = 2.0

camera.lookat[0] = 0.0

camera.lookat[1] = 0.0

camera.lookat[2] = 0.10


# ============================================================
# INITIAL POSE
# ============================================================

reset_origin()


# ============================================================
# TERMINAL
# ============================================================

print()

print(
    "===================================================="
)

print(
    "             QUADROTOR - CHALLENGE 2"
)

print(
    "===================================================="
)

print()

print("CONTROLS")

print("UP       = FORWARD")

print("DOWN     = BACKWARD")

print("LEFT     = LEFT")

print("RIGHT    = RIGHT")

print("W        = ASCEND")

print("S        = DESCEND")

print("SPACE    = STOP / LEVEL")

print("ESC      = EXIT")

print()

print("CAMERA")

print("LEFT MOUSE  = ORBIT")

print("RIGHT MOUSE = PAN")

print("MOUSE WHEEL = ZOOM")

print()

print("FRAME")

print("RED X   = FORWARD / LANE")

print("GREEN Y = LEFT / RIGHT")

print("BLUE Z  = UP")

print()

print(
    "===================================================="
)


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while not glfw.window_should_close(
        window
    ):

        # ====================================================
        # EVENTS
        # ====================================================

        glfw.poll_events()


        # ====================================================
        # CLEAR EXTERNAL FORCES
        # ====================================================

        data.xfrc_applied[:] = 0.0


        # ====================================================
        # UPDATE COMMAND
        # ====================================================

        update_command()


        # ====================================================
        # QUADROTOR CONTROLLER
        # ====================================================

        apply_quadrotor_forces()


        # ====================================================
        # PHYSICAL JITTER
        # ====================================================

        apply_physical_jitter()


        # ====================================================
        # PHYSICS
        # ====================================================

        mujoco.mj_step(
            model,
            data
        )


        # ====================================================
        # CAMERA FOLLOW
        # ====================================================

        if FOLLOW_ROBOT:

            position = data.xpos[
                base_id
            ]

            camera.lookat[0] = (
                position[0]
            )

            camera.lookat[1] = (
                position[1]
            )

            camera.lookat[2] = (
                position[2]
                + CAMERA_HEIGHT
            )


        # ====================================================
        # FRAMEBUFFER
        # ====================================================

        width, height = (
            glfw.get_framebuffer_size(
                window
            )
        )


        viewport = mujoco.MjrRect(
            0,
            0,
            width,
            height
        )


        # ====================================================
        # UPDATE SCENE
        # ====================================================

        mujoco.mjv_updateScene(

            model,

            data,

            option,

            None,

            camera,

            mujoco.mjtCatBit.mjCAT_ALL,

            scene

        )


        # ====================================================
        # WORLD FRAME
        # ====================================================

        draw_world_frame()


        # ====================================================
        # BODY FRAME
        # ====================================================

        draw_body_frame()


        # ====================================================
        # RENDER
        # ====================================================

        mujoco.mjr_render(

            viewport,

            scene,

            context

        )


        # ====================================================
        # SWAP
        # ====================================================

        glfw.swap_buffers(
            window
        )


        # ====================================================
        # TKINTER MATRIX
        # ====================================================

        update_matrix_window()

        info_window.update_idletasks()

        info_window.update()


finally:

    # ========================================================
    # STOP
    # ========================================================

    data.xfrc_applied[:] = 0.0

    data.ctrl[:] = 0.0


    # ========================================================
    # CLOSE TKINTER
    # ========================================================

    try:

        info_window.destroy()

    except Exception:

        pass


    # ========================================================
    # MUJOCO CLEANUP
    # ========================================================

    try:

        mujoco.mjr_freeContext(
            context
        )

        mujoco.mjv_freeScene(
            scene
        )

    except Exception:

        pass


    # ========================================================
    # GLFW CLEANUP
    # ========================================================

    try:

        glfw.destroy_window(
            window
        )

    except Exception:

        pass


    glfw.terminate()


print(
    "Quadrotor stopped."
)