import numpy as np
import tkinter as tk
import mujoco
import glfw


# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(
    "models/waffle_pi.xml"
)

data = mujoco.MjData(model)


# ============================================================
# FIND BASE BODY
# ============================================================

base_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base_link"
)

if base_id == -1:
    raise RuntimeError(
        "Could not find body 'base_link' in waffle_pi.xml"
    )


# ============================================================
# FIND BASE JOINTS
# ============================================================

base_x_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "base_x"
)

base_y_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "base_y"
)

base_yaw_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "base_yaw"
)

if base_x_joint == -1:
    raise RuntimeError(
        "Could not find joint 'base_x'"
    )

if base_y_joint == -1:
    raise RuntimeError(
        "Could not find joint 'base_y'"
    )

if base_yaw_joint == -1:
    raise RuntimeError(
        "Could not find joint 'base_yaw'"
    )


# ============================================================
# QPOS ADDRESSES
# ============================================================

base_x_qpos = model.jnt_qposadr[
    base_x_joint
]

base_y_qpos = model.jnt_qposadr[
    base_y_joint
]

base_yaw_qpos = model.jnt_qposadr[
    base_yaw_joint
]


# ============================================================
# DOF ADDRESSES
# ============================================================

base_x_dof = model.jnt_dofadr[
    base_x_joint
]

base_y_dof = model.jnt_dofadr[
    base_y_joint
]

base_yaw_dof = model.jnt_dofadr[
    base_yaw_joint
]


# ============================================================
# ROBOT HEIGHT
# ============================================================

INITIAL_Z = 0.11


# ============================================================
# MOTOR SPEED
# ============================================================

# Increased from the previous 6 / 4.

FORWARD_SPEED = 15.0

ROTATE_SPEED = 10.0


# ============================================================
# STARTUP / RESPAWN PHYSICAL VIBRATION
# ============================================================

STARTUP_VIBRATION_DURATION = 1.5

VIBRATION_TORQUE = 0.002

VIBRATION_FREQUENCY = 7.0

VIBRATION_NOISE = 0.0005

startup_vibration_active = True

startup_vibration_start_time = 0.0


# ============================================================
# LIVE TERMINAL MATRIX PRINT
# ============================================================

# MuJoCo timestep is 0.002 s = 500 physics steps/sec.
#
# We do NOT print 500 matrices/sec because terminal I/O
# would slow the simulation considerably.
#
# 0.05 sec = 20 matrix updates/sec.

MATRIX_PRINT_INTERVAL = 0.05

last_matrix_print_time = -1.0


# ============================================================
# CAMERA
# ============================================================

FOLLOW_ROBOT = True

CAMERA_DISTANCE = 2.0

CAMERA_ELEVATION = -20.0

CAMERA_AZIMUTH = 90.0

CAMERA_HEIGHT = 0.10


# ============================================================
# KEYBOARD STATE
# ============================================================

up_pressed = False

down_pressed = False

left_pressed = False

right_pressed = False

space_pressed = False


# ============================================================
# CURRENT CONTROL STATUS
# ============================================================

current_status = "STARTING"

current_left_command = 0.0

current_right_command = 0.0


# ============================================================
# MOUSE STATE
# ============================================================

mouse_left_pressed = False

mouse_right_pressed = False

last_mouse_x = 0.0

last_mouse_y = 0.0


# ============================================================
# MUJOCO VISUALIZATION
# ============================================================

camera = None

option = None

scene = None

context = None


# ============================================================
# WHEEL CONTROL
# ============================================================

def set_wheels(left, right):

    global current_left_command
    global current_right_command

    current_left_command = float(left)

    current_right_command = float(right)

    data.ctrl[0] = left
    data.ctrl[1] = right


def stop_robot():

    set_wheels(
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
    global space_pressed

    # --------------------------------------------------------
    # ESC
    # --------------------------------------------------------

    if key == glfw.KEY_ESCAPE:

        if action == glfw.PRESS:

            glfw.set_window_should_close(
                window,
                True
            )

        return

    # --------------------------------------------------------
    # KEY PRESS
    # --------------------------------------------------------

    if action == glfw.PRESS:

        if key == glfw.KEY_UP:

            up_pressed = True

        elif key == glfw.KEY_DOWN:

            down_pressed = True

        elif key == glfw.KEY_LEFT:

            left_pressed = True

        elif key == glfw.KEY_RIGHT:

            right_pressed = True

        elif key == glfw.KEY_SPACE:

            space_pressed = True

            stop_robot()

    # --------------------------------------------------------
    # KEY RELEASE
    # --------------------------------------------------------

    elif action == glfw.RELEASE:

        if key == glfw.KEY_UP:

            up_pressed = False

        elif key == glfw.KEY_DOWN:

            down_pressed = False

        elif key == glfw.KEY_LEFT:

            left_pressed = False

        elif key == glfw.KEY_RIGHT:

            right_pressed = False

        elif key == glfw.KEY_SPACE:

            space_pressed = False


# ============================================================
# CONTROL STATUS
# ============================================================

def get_control_status():

    if space_pressed:

        return "STOPPED - SPACE"

    if up_pressed and left_pressed:

        return "FORWARD + LEFT"

    if up_pressed and right_pressed:

        return "FORWARD + RIGHT"

    if down_pressed and left_pressed:

        return "BACKWARD + LEFT"

    if down_pressed and right_pressed:

        return "BACKWARD + RIGHT"

    if up_pressed:

        return "FORWARD"

    if down_pressed:

        return "BACKWARD"

    if left_pressed:

        return "ROTATE LEFT"

    if right_pressed:

        return "ROTATE RIGHT"

    return "STOPPED"


# ============================================================
# KEYBOARD CONTROL
# ============================================================

def update_keyboard_control():

    global current_status

    # ========================================================
    # SPACE
    # ========================================================

    if space_pressed:

        stop_robot()

        current_status = "STOPPED - SPACE"

        return

    # ========================================================
    # FORWARD / BACKWARD
    # ========================================================

    forward = 0.0

    if up_pressed:

        forward += FORWARD_SPEED

    if down_pressed:

        forward -= FORWARD_SPEED

    # ========================================================
    # TURN
    # ========================================================

    turn = 0.0

    if left_pressed:

        turn += ROTATE_SPEED

    if right_pressed:

        turn -= ROTATE_SPEED

    # ========================================================
    # DIFFERENTIAL DRIVE
    # ========================================================

    left_wheel = (
        forward - turn
    )

    right_wheel = (
        forward + turn
    )

    # ========================================================
    # LIMIT
    # ========================================================

    max_command = max(
        FORWARD_SPEED,
        ROTATE_SPEED
    )

    left_wheel = np.clip(
        left_wheel,
        -max_command,
        max_command
    )

    right_wheel = np.clip(
        right_wheel,
        -max_command,
        max_command
    )

    # ========================================================
    # SEND
    # ========================================================

    set_wheels(
        left_wheel,
        right_wheel
    )

    # ========================================================
    # STATUS
    # ========================================================

    current_status = get_control_status()


# ============================================================
# STARTUP PHYSICAL VIBRATION
# ============================================================

def apply_startup_vibration():

    global startup_vibration_active

    if not startup_vibration_active:

        return

    elapsed = (
        data.time
        - startup_vibration_start_time
    )

    # --------------------------------------------------------
    # Settling finished
    # --------------------------------------------------------

    if elapsed >= STARTUP_VIBRATION_DURATION:

        startup_vibration_active = False

        return

    # --------------------------------------------------------
    # Periodic vibration
    # --------------------------------------------------------

    phase = (
        2.0
        * np.pi
        * VIBRATION_FREQUENCY
        * data.time
    )

    periodic_component = (
        VIBRATION_TORQUE
        * np.sin(phase)
    )

    # --------------------------------------------------------
    # Small random physical component
    # --------------------------------------------------------

    random_component = np.random.normal(
        0.0,
        VIBRATION_NOISE
    )

    # --------------------------------------------------------
    # Apply to base yaw DOF
    # --------------------------------------------------------

    data.qfrc_applied[
        base_yaw_dof
    ] += (
        periodic_component
        + random_component
    )


# ============================================================
# PRINT LIVE ROTATION MATRIX
# ============================================================

def print_rotation_matrix_live():

    global last_matrix_print_time

    # --------------------------------------------------------
    # Rate limit
    # --------------------------------------------------------

    if (
        data.time
        - last_matrix_print_time
        < MATRIX_PRINT_INTERVAL
    ):

        return

    last_matrix_print_time = data.time

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    position = data.xpos[
        base_id
    ].copy()

    # --------------------------------------------------------
    # Rotation matrix
    # --------------------------------------------------------

    R_WB = data.xmat[
        base_id
    ].reshape(
        3,
        3
    ).copy()

    # --------------------------------------------------------
    # Yaw
    # --------------------------------------------------------

    yaw = np.degrees(
        np.arctan2(
            R_WB[1, 0],
            R_WB[0, 0]
        )
    )

    # --------------------------------------------------------
    # Single continuously updating terminal line
    # --------------------------------------------------------

    print(
        "\r"
        f"t={data.time:7.3f}  "
        f"X={position[0]:7.3f}  "
        f"Y={position[1]:7.3f}  "
        f"Yaw={yaw:7.3f}°  "
        f"R=["
        f"{R_WB[0,0]: .3f},"
        f"{R_WB[0,1]: .3f},"
        f"{R_WB[0,2]: .3f}; "
        f"{R_WB[1,0]: .3f},"
        f"{R_WB[1,1]: .3f},"
        f"{R_WB[1,2]: .3f}; "
        f"{R_WB[2,0]: .3f},"
        f"{R_WB[2,1]: .3f},"
        f"{R_WB[2,2]: .3f}]",
        end="",
        flush=True
    )


# ============================================================
# DRAW ARROW
# ============================================================

def draw_arrow(
    scene,
    start,
    end,
    rgba
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
            0.025,
            0.025,
            0.025
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

    mujoco.mjv_connector(

        geom,

        mujoco.mjtGeom.mjGEOM_ARROW,

        0.025,

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
# DRAW LABEL
# ============================================================

def draw_label(
    scene,
    position,
    text,
    rgba
):

    if scene.ngeom >= scene.maxgeom:

        return

    geom = scene.geoms[
        scene.ngeom
    ]

    mujoco.mjv_initGeom(

        geom,

        mujoco.mjtGeom.mjGEOM_LABEL,

        np.zeros(3),

        np.asarray(
            position,
            dtype=np.float64
        ),

        np.eye(3).flatten(),

        np.asarray(
            rgba,
            dtype=np.float32
        )
    )

    geom.label = text

    scene.ngeom += 1


# ============================================================
# DRAW WORLD FRAME
# ============================================================

def draw_world_frame(scene):

    origin = np.array([
        0.0,
        0.0,
        0.015
    ])

    axis_length = 0.60

    # ========================================================
    # WORLD X
    # ========================================================

    x_end = origin + np.array([
        axis_length,
        0.0,
        0.0
    ])

    draw_arrow(
        scene,
        origin,
        x_end,
        [1.0, 0.0, 0.0, 1.0]
    )

    draw_label(
        scene,
        x_end + np.array([
            0.06,
            0.0,
            0.03
        ]),
        "WORLD X",
        [1.0, 0.0, 0.0, 1.0]
    )

    # ========================================================
    # WORLD Y
    # ========================================================

    y_end = origin + np.array([
        0.0,
        axis_length,
        0.0
    ])

    draw_arrow(
        scene,
        origin,
        y_end,
        [0.0, 1.0, 0.0, 1.0]
    )

    draw_label(
        scene,
        y_end + np.array([
            0.0,
            0.06,
            0.03
        ]),
        "WORLD Y",
        [0.0, 1.0, 0.0, 1.0]
    )

    # ========================================================
    # WORLD Z
    # ========================================================

    z_end = origin + np.array([
        0.0,
        0.0,
        axis_length
    ])

    draw_arrow(
        scene,
        origin,
        z_end,
        [0.0, 0.3, 1.0, 1.0]
    )

    draw_label(
        scene,
        z_end + np.array([
            0.0,
            0.0,
            0.06
        ]),
        "WORLD Z",
        [0.0, 0.3, 1.0, 1.0]
    )

    # ========================================================
    # ORIGIN
    # ========================================================

    draw_label(
        scene,
        origin + np.array([
            0.05,
            0.05,
            0.05
        ]),
        "WORLD ORIGIN",
        [1.0, 1.0, 1.0, 1.0]
    )


# ============================================================
# DRAW BODY FRAME
# ============================================================

def draw_body_frame(scene):

    position = data.xpos[
        base_id
    ].copy()

    # --------------------------------------------------------
    # BODY -> WORLD
    # --------------------------------------------------------

    R_WB = data.xmat[
        base_id
    ].reshape(
        3,
        3
    ).copy()

    x_axis = R_WB[:, 0]

    y_axis = R_WB[:, 1]

    z_axis = R_WB[:, 2]

    origin = position + np.array([
        0.0,
        0.0,
        0.20
    ])

    axis_length = 0.50

    # ========================================================
    # BODY X
    # ========================================================

    x_end = (
        origin
        + axis_length * x_axis
    )

    draw_arrow(
        scene,
        origin,
        x_end,
        [1.0, 0.0, 0.0, 1.0]
    )

    draw_label(
        scene,
        x_end + np.array([
            0.04,
            0.0,
            0.04
        ]),
        "BODY X / FORWARD",
        [1.0, 0.0, 0.0, 1.0]
    )

    # ========================================================
    # BODY Y
    # ========================================================

    y_end = (
        origin
        + axis_length * y_axis
    )

    draw_arrow(
        scene,
        origin,
        y_end,
        [0.0, 1.0, 0.0, 1.0]
    )

    draw_label(
        scene,
        y_end + np.array([
            0.0,
            0.04,
            0.04
        ]),
        "BODY Y / LEFT",
        [0.0, 1.0, 0.0, 1.0]
    )

    # ========================================================
    # BODY Z
    # ========================================================

    z_end = (
        origin
        + axis_length * z_axis
    )

    draw_arrow(
        scene,
        origin,
        z_end,
        [0.0, 0.3, 1.0, 1.0]
    )

    draw_label(
        scene,
        z_end + np.array([
            0.0,
            0.0,
            0.06
        ]),
        "BODY Z / UP",
        [0.0, 0.3, 1.0, 1.0]
    )

    # ========================================================
    # BODY ORIGIN
    # ========================================================

    draw_label(
        scene,
        origin + np.array([
            0.04,
            0.04,
            0.05
        ]),
        "BODY ORIGIN",
        [1.0, 1.0, 1.0, 1.0]
    )


# ============================================================
# RESET ORIGIN
# ============================================================

def reset_origin():

    global startup_vibration_active
    global startup_vibration_start_time
    global current_status

    stop_robot()

    # --------------------------------------------------------
    # Restart physical settling
    # --------------------------------------------------------

    startup_vibration_active = True

    startup_vibration_start_time = data.time

    # --------------------------------------------------------
    # Clear forces
    # --------------------------------------------------------

    data.qfrc_applied[:] = 0.0

    # --------------------------------------------------------
    # Set pose
    # --------------------------------------------------------

    data.qpos[
        base_x_qpos
    ] = 0.0

    data.qpos[
        base_y_qpos
    ] = 0.0

    data.qpos[
        base_yaw_qpos
    ] = 0.0

    # --------------------------------------------------------
    # Clear velocities
    # --------------------------------------------------------

    data.qvel[:] = 0.0

    # --------------------------------------------------------
    # Clear controls
    # --------------------------------------------------------

    data.ctrl[:] = 0.0

    # --------------------------------------------------------
    # Recalculate
    # --------------------------------------------------------

    mujoco.mj_forward(
        model,
        data
    )

    current_status = (
        "RESPAWNED AT ORIGIN"
    )

    update_spawn_entries(
        0.0,
        0.0,
        0.0
    )


# ============================================================
# SPAWN USER POSE
# ============================================================

def spawn_at_coordinates(
    x,
    y,
    yaw_degrees
):

    global startup_vibration_active
    global startup_vibration_start_time
    global current_status

    stop_robot()

    # --------------------------------------------------------
    # Restart settling
    # --------------------------------------------------------

    startup_vibration_active = True

    startup_vibration_start_time = data.time

    # --------------------------------------------------------
    # Convert yaw
    # --------------------------------------------------------

    yaw = np.radians(
        yaw_degrees
    )

    # --------------------------------------------------------
    # Set pose
    # --------------------------------------------------------

    data.qpos[
        base_x_qpos
    ] = x

    data.qpos[
        base_y_qpos
    ] = y

    data.qpos[
        base_yaw_qpos
    ] = yaw

    # --------------------------------------------------------
    # Clear velocity
    # --------------------------------------------------------

    data.qvel[:] = 0.0

    # --------------------------------------------------------
    # Clear controls
    # --------------------------------------------------------

    data.ctrl[:] = 0.0

    # --------------------------------------------------------
    # Clear forces
    # --------------------------------------------------------

    data.qfrc_applied[:] = 0.0

    # --------------------------------------------------------
    # Recalculate immediately
    # --------------------------------------------------------

    mujoco.mj_forward(
        model,
        data
    )

    current_status = "RESPAWNED"


# ============================================================
# RANDOM SPAWN
# ============================================================

def random_spawn():

    random_x = np.random.uniform(
        -10.0,
        10.0
    )

    random_y = np.random.uniform(
        -1.5,
        1.5
    )

    random_yaw = np.random.uniform(
        -180.0,
        180.0
    )

    spawn_at_coordinates(
        random_x,
        random_y,
        random_yaw
    )

    update_spawn_entries(
        random_x,
        random_y,
        random_yaw
    )


# ============================================================
# UPDATE SPAWN INPUTS
# ============================================================

def update_spawn_entries(
    x,
    y,
    yaw
):

    x_entry.delete(
        0,
        tk.END
    )

    x_entry.insert(
        0,
        f"{x:.3f}"
    )

    y_entry.delete(
        0,
        tk.END
    )

    y_entry.insert(
        0,
        f"{y:.3f}"
    )

    yaw_entry.delete(
        0,
        tk.END
    )

    yaw_entry.insert(
        0,
        f"{yaw:.3f}"
    )


# ============================================================
# APPLY USER POSE
# ============================================================

def apply_user_pose():

    try:

        x = float(
            x_entry.get()
        )

        y = float(
            y_entry.get()
        )

        yaw = float(
            yaw_entry.get()
        )

        spawn_at_coordinates(
            x,
            y,
            yaw
        )

    except ValueError:

        print(
            "\nERROR: Please enter valid numbers."
        )


# ============================================================
# CAMERA FOLLOW
# ============================================================

def update_follow_camera():

    if not FOLLOW_ROBOT:

        return

    position = data.xpos[
        base_id
    ]

    camera.lookat[0] = position[0]

    camera.lookat[1] = position[1]

    camera.lookat[2] = (
        position[2]
        + CAMERA_HEIGHT
    )


# ============================================================
# MOUSE BUTTON
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

    dx = xpos - last_mouse_x

    dy = ypos - last_mouse_y

    last_mouse_x = xpos

    last_mouse_y = ypos

    # --------------------------------------------------------
    # LEFT DRAG
    # --------------------------------------------------------

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
            -5.0
        )

        camera.azimuth = (
            CAMERA_AZIMUTH
        )

        camera.elevation = (
            CAMERA_ELEVATION
        )

    # --------------------------------------------------------
    # RIGHT DRAG
    # --------------------------------------------------------

    elif mouse_right_pressed:

        if not FOLLOW_ROBOT:

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
        yoffset * 0.15
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
# ROTATION MATRIX WINDOW
# ============================================================

info_window = tk.Tk()

info_window.title(
    "TurtleBot3 Waffle Pi - Robot Pose & Rotation Matrix"
)

info_window.geometry(
    "680x850"
)

info_window.minsize(
    620,
    720
)


# ============================================================
# SPAWN FRAME
# ============================================================

spawn_frame = tk.LabelFrame(
    info_window,
    text="RESPAWN / SET ROBOT POSE"
)

spawn_frame.pack(
    padx=10,
    pady=8,
    fill=tk.X
)


# ============================================================
# X
# ============================================================

tk.Label(
    spawn_frame,
    text="X (m)"
).grid(
    row=0,
    column=0,
    padx=5,
    pady=5
)

x_entry = tk.Entry(
    spawn_frame,
    width=12
)

x_entry.insert(
    0,
    "0.000"
)

x_entry.grid(
    row=0,
    column=1,
    padx=5
)


# ============================================================
# Y
# ============================================================

tk.Label(
    spawn_frame,
    text="Y (m)"
).grid(
    row=0,
    column=2,
    padx=5,
    pady=5
)

y_entry = tk.Entry(
    spawn_frame,
    width=12
)

y_entry.insert(
    0,
    "0.000"
)

y_entry.grid(
    row=0,
    column=3,
    padx=5
)


# ============================================================
# Z
# ============================================================

tk.Label(
    spawn_frame,
    text="Z (m)"
).grid(
    row=1,
    column=0,
    padx=5,
    pady=5
)

z_entry = tk.Entry(
    spawn_frame,
    width=12,
    state="readonly"
)

z_entry.configure(
    state="normal"
)

z_entry.insert(
    0,
    f"{INITIAL_Z:.3f}"
)

z_entry.configure(
    state="readonly"
)

z_entry.grid(
    row=1,
    column=1,
    padx=5
)


# ============================================================
# YAW
# ============================================================

tk.Label(
    spawn_frame,
    text="Yaw (deg)"
).grid(
    row=1,
    column=2,
    padx=5,
    pady=5
)

yaw_entry = tk.Entry(
    spawn_frame,
    width=12
)

yaw_entry.insert(
    0,
    "0.000"
)

yaw_entry.grid(
    row=1,
    column=3,
    padx=5
)


# ============================================================
# APPLY
# ============================================================

apply_button = tk.Button(
    spawn_frame,
    text="APPLY POSE",
    command=apply_user_pose,
    width=14,
    height=2
)

apply_button.grid(
    row=2,
    column=0,
    columnspan=2,
    padx=5,
    pady=8
)


# ============================================================
# RESET
# ============================================================

reset_button = tk.Button(
    spawn_frame,
    text="RESET ORIGIN",
    command=reset_origin,
    width=14,
    height=2
)

reset_button.grid(
    row=2,
    column=2,
    padx=5,
    pady=8
)


# ============================================================
# RANDOM
# ============================================================

random_button = tk.Button(
    spawn_frame,
    text="RANDOM SPAWN",
    command=random_spawn,
    width=14,
    height=2
)

random_button.grid(
    row=2,
    column=3,
    padx=5,
    pady=8
)


# ============================================================
# CAMERA FRAME
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

    FOLLOW_ROBOT = follow_var.get()


follow_check = tk.Checkbutton(
    camera_frame,
    text="FOLLOW ROBOT",
    variable=follow_var,
    command=toggle_follow
)

follow_check.pack(
    side=tk.LEFT,
    padx=10
)

tk.Label(
    camera_frame,
    text=(
        "Left drag: Orbit   "
        "Right drag: Pan   "
        "Wheel: Zoom"
    )
).pack(
    side=tk.LEFT,
    padx=5
)


# ============================================================
# MATRIX DISPLAY
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
    font=("Courier New", 12),
    width=68,
    height=32,
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
# UPDATE MATRIX WINDOW
# ============================================================

def update_matrix_window():

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    position = data.xpos[
        base_id
    ].copy()

    # --------------------------------------------------------
    # R_WB
    # --------------------------------------------------------

    R_WB = data.xmat[
        base_id
    ].reshape(
        3,
        3
    ).copy()

    # --------------------------------------------------------
    # R_BW
    # --------------------------------------------------------

    R_BW = R_WB.T

    # --------------------------------------------------------
    # Yaw
    # --------------------------------------------------------

    yaw = np.degrees(
        np.arctan2(
            R_WB[1, 0],
            R_WB[0, 0]
        )
    )

    # --------------------------------------------------------
    # Build display
    # --------------------------------------------------------

    text = ""

    text += (
        "====================================================\n"
    )

    text += (
        "          TURTLEBOT3 WAFFLE PI\n"
    )

    text += (
        "          LIVE ROBOT POSE\n"
    )

    text += (
        "====================================================\n\n"
    )

    # ========================================================
    # POSITION
    # ========================================================

    text += "POSITION\n"

    text += (
        "----------------------------------------------------\n"
    )

    text += (
        f"X       = {position[0]: .3f} m\n"
    )

    text += (
        f"Y       = {position[1]: .3f} m\n"
    )

    text += (
        f"Z       = {position[2]: .3f} m\n"
    )

    text += (
        f"YAW     = {yaw: .3f} deg\n\n"
    )

    # ========================================================
    # R_WB
    # ========================================================

    text += (
        "R_WB : BODY -> WORLD\n"
    )

    text += (
        "----------------------------------------------------\n"
    )

    text += (
        f"[ {R_WB[0,0]: .3f}  "
        f"{R_WB[0,1]: .3f}  "
        f"{R_WB[0,2]: .3f} ]\n"
    )

    text += (
        f"[ {R_WB[1,0]: .3f}  "
        f"{R_WB[1,1]: .3f}  "
        f"{R_WB[1,2]: .3f} ]\n"
    )

    text += (
        f"[ {R_WB[2,0]: .3f}  "
        f"{R_WB[2,1]: .3f}  "
        f"{R_WB[2,2]: .3f} ]\n\n"
    )

    # ========================================================
    # R_BW
    # ========================================================

    text += (
        "R_BW : WORLD -> BODY\n"
    )

    text += (
        "----------------------------------------------------\n"
    )

    text += (
        f"[ {R_BW[0,0]: .3f}  "
        f"{R_BW[0,1]: .3f}  "
        f"{R_BW[0,2]: .3f} ]\n"
    )

    text += (
        f"[ {R_BW[1,0]: .3f}  "
        f"{R_BW[1,1]: .3f}  "
        f"{R_BW[1,2]: .3f} ]\n"
    )

    text += (
        f"[ {R_BW[2,0]: .3f}  "
        f"{R_BW[2,1]: .3f}  "
        f"{R_BW[2,2]: .3f} ]\n\n"
    )

    # ========================================================
    # BODY AXES
    # ========================================================

    text += (
        "BODY AXES IN WORLD FRAME\n"
    )

    text += (
        "----------------------------------------------------\n"
    )

    text += (
        f"BODY X = "
        f"({R_WB[0,0]: .3f}, "
        f"{R_WB[1,0]: .3f}, "
        f"{R_WB[2,0]: .3f})\n"
    )

    text += (
        f"BODY Y = "
        f"({R_WB[0,1]: .3f}, "
        f"{R_WB[1,1]: .3f}, "
        f"{R_WB[2,1]: .3f})\n"
    )

    text += (
        f"BODY Z = "
        f"({R_WB[0,2]: .3f}, "
        f"{R_WB[1,2]: .3f}, "
        f"{R_WB[2,2]: .3f})\n\n"
    )

    # ========================================================
    # STATUS
    # ========================================================

    text += "STATUS\n"

    text += (
        "----------------------------------------------------\n"
    )

    text += (
        f"{current_status}\n"
    )

    text += (
        f"Left Wheel  = "
        f"{current_left_command:.3f}\n"
    )

    text += (
        f"Right Wheel = "
        f"{current_right_command:.3f}\n"
    )

    if startup_vibration_active:

        elapsed = (
            data.time
            - startup_vibration_start_time
        )

        remaining = max(
            0.0,
            STARTUP_VIBRATION_DURATION
            - elapsed
        )

        text += "\n"

        text += (
            f"STARTUP SETTLING = "
            f"{remaining:.3f} s\n"
        )

    else:

        text += "\n"

        text += (
            "STARTUP VIBRATION = OFF\n"
        )

    # ========================================================
    # SIMULATION TIME
    # ========================================================

    text += "\n"

    text += (
        f"Simulation Time = "
        f"{data.time:.3f} s\n"
    )

    # ========================================================
    # UPDATE
    # ========================================================

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
# CREATE MUJOCO WINDOW
# ============================================================

window = glfw.create_window(
    1400,
    900,
    "TurtleBot3 Waffle Pi - MuJoCo",
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
# CAMERA
# ============================================================

camera.azimuth = CAMERA_AZIMUTH

camera.elevation = CAMERA_ELEVATION

camera.distance = CAMERA_DISTANCE

camera.lookat[0] = 0.0

camera.lookat[1] = 0.0

camera.lookat[2] = INITIAL_Z


# ============================================================
# INITIAL PHYSICS
# ============================================================

mujoco.mj_forward(
    model,
    data
)


# ============================================================
# STARTUP VIBRATION TIMER
# ============================================================

startup_vibration_start_time = data.time


# ============================================================
# INITIAL STATUS
# ============================================================

current_status = (
    "STARTUP SETTLING"
)


# ============================================================
# TERMINAL START MESSAGE
# ============================================================

print()

print(
    "================================================"
)

print(
    "       TURTLEBOT3 WAFFLE PI - MUJOCO"
)

print(
    "================================================"
)

print()

print("CONTROLS")

print("UP           = Forward")

print("DOWN         = Backward")

print("LEFT         = Rotate Left")

print("RIGHT        = Rotate Right")

print("UP + LEFT    = Forward + Left")

print("UP + RIGHT   = Forward + Right")

print("DOWN + LEFT  = Backward + Left")

print("DOWN + RIGHT = Backward + Right")

print("SPACE        = Stop")

print("ESC          = Exit")

print()

print("SPEED")

print(
    f"Forward Speed = {FORWARD_SPEED:.3f}"
)

print(
    f"Rotate Speed  = {ROTATE_SPEED:.3f}"
)

print()

print("CAMERA")

print("Left Mouse  = Orbit")

print("Right Mouse = Pan")

print("Wheel       = Zoom")

print()

print(
    "Rotation matrix is printed live at 20 Hz."
)

print()

print(
    "================================================"
)


# ============================================================
# MAIN LOOP
# ============================================================

while not glfw.window_should_close(
    window
):

    # ========================================================
    # EVENTS
    # ========================================================

    glfw.poll_events()


    # ========================================================
    # CLEAR EXTERNAL FORCES
    # ========================================================

    data.qfrc_applied[:] = 0.0


    # ========================================================
    # STARTUP / RESPAWN VIBRATION
    # ========================================================

    apply_startup_vibration()


    # ========================================================
    # KEYBOARD CONTROL
    # ========================================================

    update_keyboard_control()


    # ========================================================
    # PHYSICS
    # ========================================================

    mujoco.mj_step(
        model,
        data
    )


    # ========================================================
    # PRINT ACTUAL MATRIX AFTER PHYSICS STEP
    # ========================================================

    print_rotation_matrix_live()


    # ========================================================
    # CAMERA FOLLOW
    # ========================================================

    update_follow_camera()


    # ========================================================
    # FRAMEBUFFER
    # ========================================================

    width, height = glfw.get_framebuffer_size(
        window
    )

    viewport = mujoco.MjrRect(
        0,
        0,
        width,
        height
    )


    # ========================================================
    # UPDATE MUJOCO SCENE
    # ========================================================

    mujoco.mjv_updateScene(
        model,
        data,
        option,
        None,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene
    )


    # ========================================================
    # WORLD FRAME
    # ========================================================

    draw_world_frame(
        scene
    )


    # ========================================================
    # BODY FRAME
    # ========================================================

    draw_body_frame(
        scene
    )


    # ========================================================
    # RENDER
    # ========================================================

    mujoco.mjr_render(
        viewport,
        scene,
        context
    )


    # ========================================================
    # SWAP
    # ========================================================

    glfw.swap_buffers(
        window
    )


    # ========================================================
    # UPDATE MATRIX WINDOW
    # ========================================================

    update_matrix_window()

    info_window.update_idletasks()

    info_window.update()


# ============================================================
# STOP ROBOT
# ============================================================

stop_robot()


# ============================================================
# CLEANUP
# ============================================================

try:

    info_window.destroy()

except Exception:

    pass


mujoco.mjr_freeContext(
    context
)

mujoco.mjv_freeScene(
    scene
)

glfw.destroy_window(
    window
)

glfw.terminate()