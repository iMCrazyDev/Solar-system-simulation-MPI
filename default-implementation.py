from vpython import *
from math import sin, cos, radians

G = 6.67430e-11
dt = 60 * 60 * 2  # 2 hours per simulation step
scale = 1e9
camera_distance = 4.5e11 / scale

planet_data = [
    ("Sun",     1.989e30, 7e8,    0,        0,     color.white,    0.0),
    ("Mercury", 3.30e23,  2.44e6, 5.79e10,  47400, color.gray(0.5), 7.0),
    ("Venus",   4.87e24,  6.05e6, 1.08e11,  35000, color.orange,    3.39),
    ("Earth",   5.97e24,  6.37e6, 1.50e11,  29780, color.blue,      0.0),
    ("Mars",    6.42e23,  3.39e6, 2.28e11,  24070, color.red,       1.85),
    ("Jupiter", 1.90e27,  6.99e7, 7.78e11,  13070, color.orange,    1.30),
    ("Saturn",  5.68e26,  5.82e7, 1.43e12,   9700, color.yellow,    2.49),
    ("Uranus",  8.68e25,  2.54e7, 2.87e12,   6800, color.cyan,      0.77),
    ("Neptune", 1.02e26,  2.46e7, 4.50e12,   5400, color.blue,      1.77),
]

scene.title = "Solar System with Rocket Simulation"
scene.forward = vector(-1, -0.3, -1)
scene.range = 1e12 / scale
scene.center = vector(0, 0, 0)

bodies = []
focused_body = None

# Create planetary bodies
for name, mass, radius, distance, speed, col, inc in planet_data:
    inc_rad = radians(inc)
    pos = vector(distance / scale * cos(inc_rad),
                 distance / scale * sin(inc_rad), 0)
    vel = vector(-speed * sin(inc_rad), speed * cos(inc_rad), 0)

    body = sphere(pos=pos, radius=radius * 8 / scale,
                  color=col, make_trail=True, trail_color=col, retain=300)
    body.mass = mass
    body.velocity = vel
    body.name = name
    bodies.append(body)

# Set total system momentum to zero
total_momentum = sum((b.mass * b.velocity for b in bodies[1:]), vector(0, 0, 0))
bodies[0].velocity = -total_momentum / bodies[0].mass

# Camera control keys
planet_keys = {'0': 'Sun', '1': 'Mercury', '2': 'Venus', '3': 'Earth',
               '4': 'Mars', '5': 'Jupiter', '6': 'Saturn',
               '7': 'Uranus', '8': 'Neptune'}

def set_focus(obj):
    global focused_body
    focused_body = obj
    print(f"Focusing on: {obj.name}")

def key_input(evt):
    global focused_body
    key = evt.key
    if key in planet_keys:
        name = planet_keys[key]
        for b in bodies:
            if b.name == name:
                set_focus(b)
                break
    elif key == '9':
        focused_body = None

scene.bind('keydown', key_input)

# Convert color vector to hex string for UI labels
def to_hex(c):
    return '#{:02x}{:02x}{:02x}'.format(int(c.x * 255), int(c.y * 255), int(c.z * 255))

# UI for adjusting planetary masses
initial_masses = {b.name: b.mass for b in bodies}
mass_labels = {}

wtext(text="\nMass settings:\n")
for b in bodies:
    color_hex = to_hex(b.color)
    wtext(text=f"<font color='{color_hex}'>{b.name}</font>: ")
    def make_slider(body):
        def update(s):
            body.mass = s.value * initial_masses[body.name]
            mass_labels[body.name].text = f"{body.mass:.2e} kg\n"
        return update
    slider(min=0.1, max=100.0, value=1.0, length=200, bind=make_slider(b), right=15)
    mass_labels[b.name] = wtext(text=f"{b.mass:.2e} kg\n")

# Simulation speed control
sim_speed = {'value': 300}
wtext(text="\nSimulation speed:\n")
def update_rate(s):
    sim_speed['value'] = int(s.value)
    rate_label.text = f"Rate: {int(s.value)} steps/sec\n"
slider(min=10, max=1000, value=300, length=300, step=10, bind=update_rate, right=15)
rate_label = wtext(text=f"Rate: {sim_speed['value']} steps/sec\n")

# Rocket speed display
wtext(text="\nRocket speed:\n")
rocket_speed_text = wtext(text="0.00 km/s")

# Elapsed simulation time display
wtext(text="\nElapsed time:\n")
time_elapsed_text = wtext(text="0 d 00:00")

# Rocket initialization
earth = next(b for b in bodies if b.name == "Earth")
rocket = sphere(pos=earth.pos + vector(1e7 / scale, 0, 0), radius=5e6 / scale,
                color=color.green, make_trail=True, retain=500)
rocket.mass = 5e5
rocket.initial_mass = 5e5
rocket.dry_mass = 1e5
rocket.velocity = earth.velocity

# Rocket flight plan with thrust phases
flight_plan = [
    {
        "start_time": 60 * 60 * 24,
        "duration":   60 * 60 * 6,
        "acceleration": vector(0.1, -1, 0)
    },
    {
        "start_time": 60 * 60 * 72,
        "duration":   60 * 60 * 4,
        "acceleration": vector(0.0003, 0.0002, 0)
    }
]

flight_index = 0
step = 0
time_passed = 0

# Main simulation loop
while True:
    rate(sim_speed['value'])
    time_passed += dt

    all_bodies = bodies + [rocket]
    forces = [vector(0, 0, 0) for _ in all_bodies]

    for i, bi in enumerate(all_bodies):
        for j, bj in enumerate(all_bodies):
            if i == j:
                continue
            r_vec = bj.pos - bi.pos
            r_mag = mag(r_vec)
            r_hat = norm(r_vec)
            f = G * bi.mass * bj.mass / (r_mag * scale)**2
            forces[i] += f * r_hat

    for i, b in enumerate(bodies):
        acc = forces[i] / b.mass
        b.velocity += acc * dt
        b.pos += b.velocity * dt / scale

    rocket_acc = forces[-1] / rocket.mass
    if flight_index < len(flight_plan):
        phase = flight_plan[flight_index]
        t0 = phase["start_time"]
        dur = phase["duration"]
        acc_vec = phase["acceleration"]
        if t0 <= time_passed < t0 + dur:
            rocket_acc += acc_vec
            burn_rate = (rocket.initial_mass - rocket.dry_mass) / dur
            rocket.mass = max(rocket.mass - burn_rate * dt, rocket.dry_mass)
        elif time_passed >= t0 + dur:
            flight_index += 1

    rocket.velocity += rocket_acc * dt
    rocket.pos += rocket.velocity * dt / scale

    if focused_body:
        direction = norm(vector(-1, -1, -0.5))
        scene.camera.pos = focused_body.pos + direction * camera_distance
        scene.camera.axis = focused_body.pos - scene.camera.pos

    rocket_speed = mag(rocket.velocity) / 1000
    rocket_speed_text.text = f"{rocket_speed:.2f} km/s"

    days = int(time_passed // (60 * 60 * 24))
    hours = int((time_passed % (60 * 60 * 24)) // 3600)
    minutes = int((time_passed % 3600) // 60)
    time_elapsed_text.text = f"{days} d {hours:02}:{minutes:02}"
