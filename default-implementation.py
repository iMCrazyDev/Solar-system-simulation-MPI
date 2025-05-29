# Modified Solar System Simulation with simulation-time display
from vpython import *
from math import sin, cos, radians
import threading, time

G           = 6.67430e-11
DT          = 60 * 2          # 2-минутный физический шаг
RENDER_FPS  = 60              # частота отрисовки
SCALE       = 1e9
CAMERA_DIST = 4.5e11 / SCALE

planet_data = [
    ("Sun",     1.989e30, 7e8,    0,        0,        color.white,    0.0),
    ("Mercury", 3.30e23,  2.44e6, 5.79e10,  47400,    color.gray(0.5), 7.0),
    ("Venus",   4.87e24,  6.05e6, 1.08e11,  35000,    color.orange,    3.39),
    ("Earth",   5.97e24,  6.37e6, 1.50e11,  29780,    color.blue,      0.0),
    ("Mars",    6.42e23,  3.39e6, 2.28e11,  24070,    color.red,       1.85),
    ("Jupiter", 1.90e27,  6.99e7, 7.78e11,  13070,    color.orange,    1.30),
    ("Saturn",  5.68e26,  5.82e7, 1.43e12,   9700,    color.yellow,    2.49),
    ("Uranus",  8.68e25,  2.54e7, 2.87e12,   6800,    color.cyan,      0.77),
    ("Neptune", 1.02e26,  2.46e7, 4.50e12,   5400,    color.blue,      1.77),
]

scene.title   = "Solar System Simulation"
scene.forward = vector(-1, -0.3, -1)
scene.range   = 1e12 / SCALE
scene.center  = vector(0, 0, 0)

bodies = []
for name, mass, rad, a, v, col, inc in planet_data:
    inc = radians(inc)
    pos = vector(a * cos(inc), a * sin(inc), 0)
    vel = vector(-v * sin(inc), v * cos(inc), 0)

    s = sphere(pos=pos/SCALE, radius=rad*8/SCALE,
               color=col, make_trail=True, trail_color=col, retain=300)
    s.name      = name
    s.mass      = mass
    s.real_pos  = pos
    s.velocity  = vel
    s.acc       = vector(0, 0, 0)
    bodies.append(s)

total_p = sum((b.mass*b.velocity for b in bodies[1:]), vector(0,0,0))
bodies[0].velocity = -total_p / bodies[0].mass

shared_state = {
    "positions":  [b.real_pos for b in bodies],
    "velocities": [b.velocity  for b in bodies],
    "phys_fps":   0.0,
    "sim_time":   0.0    
}

def physics_thread():
    for i, bi in enumerate(bodies):
        F = vector(0,0,0)
        for j, bj in enumerate(bodies):
            if i == j: continue
            r = bj.real_pos - bi.real_pos
            F += G * bi.mass * bj.mass * norm(r) / mag2(r)
        bi.acc = F / bi.mass

    steps, t_mark = 0, time.time()
    sim_time = 0.0
    while True:
        for b in bodies:
            b.velocity += 0.5 * b.acc * DT
        for b in bodies:
            b.real_pos += b.velocity * DT

        for i, bi in enumerate(bodies):
            F = vector(0,0,0)
            for j, bj in enumerate(bodies):
                if i == j: continue
                r = bj.real_pos - bi.real_pos
                F += G * bi.mass * bj.mass * norm(r) / mag2(r)
            bi.acc = F / bi.mass
        for b in bodies:
            b.velocity += 0.5 * b.acc * DT

        shared_state["positions"]  = [b.real_pos for b in bodies]
        shared_state["velocities"] = [b.velocity  for b in bodies]

        sim_time += DT
        shared_state["sim_time"] = sim_time

        steps += 1
        now = time.time()
        if now - t_mark >= 1.0:
            shared_state["phys_fps"] = steps / (now - t_mark)
            steps, t_mark = 0, now

threading.Thread(target=physics_thread, daemon=True).start()

fps_label  = wtext(text="Physics FPS: 0.0\n")
time_label = wtext(text="Simulation time: 0.0 days\n")

while True:
    rate(RENDER_FPS)
    for body, pos in zip(bodies, shared_state["positions"]):
        body.pos = pos / SCALE

    fps_label.text  = f"Physics FPS: {shared_state['phys_fps']:.1f}\n"
    days = shared_state["sim_time"] / 86400.0
    time_label.text = f"Simulation time: {days:.2f} days\n"

    if scene.camera.follow is None:
        scene.camera.pos  = vector(0, 0, CAMERA_DIST / SCALE)
        scene.camera.axis = vector(0, 0, -CAMERA_DIST / SCALE)
