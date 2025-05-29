#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
from math import radians, sin, cos
import time, threading, signal

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
size  = comm.Get_size()

G          = 6.67430e-11
DT         = 60 * 2          # физический шаг, 2 минуты
SCALE      = 1e9
RENDER_FPS = 60              # кадров/с GUI

planet_data = [
    ("Sun",     1.989e30, 7.00e8,  0.00e0,   0,       (1,1,1),     0.00),
    ("Mercury", 3.30e23,  2.44e6,  5.79e10,  47400,   (0.5,0.5,0.5),7.00),
    ("Venus",   4.87e24,  6.05e6,  1.08e11,  35000,   (1,0.6,0),    3.39),
    ("Earth",   5.97e24,  6.37e6,  1.50e11,  29780,   (0.1,0.3,1),  0.00),
    ("Mars",    6.42e23,  3.39e6,  2.28e11,  24070,   (1,0,0),      1.85),
    ("Jupiter", 1.90e27,  6.99e7,  7.78e11,  13070,   (1,0.6,0.2),  1.30),
    ("Saturn",  5.68e26,  5.82e7,  1.43e12,  9700,    (1,1,0),      2.49),
    ("Uranus",  8.68e25,  2.54e7,  2.87e12,  6800,    (0,1,1),      0.77),
    ("Neptune", 1.02e26,  2.46e7,  4.50e12,  5400,    (0,0.2,1),    1.77),
]
N = len(planet_data)

# ────────────────── arrays ──────────────────
masses     = np.empty(N)
positions  = np.empty((N, 3))
velocities = np.empty((N, 3))
radii      = np.empty(N)
colors     = []

for i, (_, m, r, a, v, col, inc) in enumerate(planet_data):
    inc = radians(inc)
    masses[i]     = m
    radii[i]      = r
    positions[i]  = (a*cos(inc), a*sin(inc), 0)
    velocities[i] = (-v*sin(inc), v*cos(inc), 0)
    colors.append(col)

velocities -= np.sum(masses[:, None] * velocities, axis=0) / masses.sum()

counts = np.full(size, N//size, dtype=int)
counts[: N % size] += 1
starts   = np.insert(np.cumsum(counts)[:-1], 0, 0)
local    = slice(starts[rank], starts[rank] + counts[rank])

if rank == 0:
    lock     = threading.Lock()
    gui_pos  = positions.copy()
    gui_fps  = [0.0]

if rank == 0:
    fps_samples = []       
    stats_done  = False
    t_start     = time.time()

forces = np.zeros_like(positions)

def physics_loop():
    global positions, velocities, forces
    steps_1s, tic_1s = 0, time.time()

    while True:
        loop_t0 = time.time()
        forces_local = np.zeros_like(forces)
        for i in range(local.start, local.stop):
            rij   = positions - positions[i]
            dist2 = np.einsum("ij,ij->i", rij, rij)
            mask  = dist2 > 0
            inv_r3 = np.zeros_like(dist2)
            inv_r3[mask] = 1.0 / np.power(dist2[mask], 1.5)
            coef  = G * masses[i] * masses * inv_r3
            forces_local[i] = (rij * coef[:, None]).sum(axis=0)
        comm.Allreduce(forces_local, forces, op=MPI.SUM)

        # 2) Velocity‑Verlet
        velocities += 0.5 * (forces / masses[:, None]) * DT
        positions  += velocities * DT

        # 3) recompute forces for new positions
        forces_local.fill(0)
        for i in range(local.start, local.stop):
            rij   = positions - positions[i]
            dist2 = np.einsum("ij,ij->i", rij, rij)
            mask  = dist2 > 0
            inv_r3 = np.zeros_like(dist2)
            inv_r3[mask] = 1.0 / np.power(dist2[mask], 1.5)
            coef  = G * masses[i] * masses * inv_r3
            forces_local[i] = (rij * coef[:, None]).sum(axis=0)
        comm.Allreduce(forces_local, forces, op=MPI.SUM)
        velocities += 0.5 * (forces / masses[:, None]) * DT

        loop_dt = time.time() - loop_t0

        # ── rank 0: FPS bookkeeping ──
        if rank == 0:
            fps = 1.0 / loop_dt if loop_dt > 0 else 0.0
            fps_samples.append(fps)
            steps_1s += 1
            if time.time() - tic_1s >= 1.0:
                with lock:
                    gui_fps[0] = steps_1s / (time.time() - tic_1s)
                steps_1s, tic_1s = 0, time.time()
                with lock:
                    gui_pos[:] = positions

            global stats_done
            if not stats_done and time.time() - t_start >= 60.0:
                samples = np.array(fps_samples)
                samples.sort()
                n = len(samples)
                avg = samples.mean()
                low1pct   = samples[: max(1, int(n * 0.01))].mean()
                low0_1pct = samples[: max(1, int(n * 0.001))].mean()
                print("\n=== 60‑second FPS statistics ===")
                print(f"Avg FPS     : {avg:10.1f}")
                print(f"1 % Low FPS  : {low1pct:10.1f}")
                print(f"0.1 % Low FPS: {low0_1pct:10.1f}\n")
                stats_done = True

if rank == 0:
    def gui_thread():
        import signal as _sig, threading as _th
        if _th.current_thread() is not _th.main_thread():
            _sig.signal = lambda *a, **k: None
        from vpython import sphere, vector, scene, rate, wtext

        scene.title   = "Solar System — MPI + threaded GUI"
        scene.range   = 1e12 / SCALE
        scene.forward = vector(-1, -0.3, -1)
        scene.center  = vector(0, 0, 0)
        spheres = [
            sphere(pos=vector(*(gui_pos[i] / SCALE)), radius=radii[i] * 8 / SCALE,
                   color=vector(*colors[i]), make_trail=True, retain=300)
            for i in range(N)
        ]
        fps_lbl = wtext(text="\nPhysics FPS: 0\n")
        while True:
            rate(RENDER_FPS)
            with lock:
                pos_cp = gui_pos.copy(); fps_now = gui_fps[0]
            for s, p in zip(spheres, pos_cp):
                s.pos = vector(*(p / SCALE))
            fps_lbl.text = f"Physics FPS: {fps_now:.1f}\n"

    threading.Thread(target=gui_thread, daemon=True).start()
    physics_loop()
else:
    physics_loop()
