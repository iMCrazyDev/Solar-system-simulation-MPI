#!/usr/bin/env python3
"""
Solar‑system N‑body — **MPI benchmark, one‑line result**
───────────────────────────────────────────────────────
• Прогрев `--warmup` (default 10 s)
• Измерение `--secs` (default 30 s)
• **Единичный** агрегированный вывод (rank 0) после измерения.

Запуск:
    mpiexec -n 8 python solar_mpi_bench.py --warmup 10 --secs 30
"""
from __future__ import annotations
import argparse, time
from math import radians, sin, cos
import numpy as np
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument("--warmup", type=float, default=10.0)
parser.add_argument("--secs",   type=float, default=30.0)
args = parser.parse_args()
WARMUP, BENCH = args.warmup, args.secs

comm, rank, size = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()

G, DT = 6.67430e-11, 120.0  # 2‑min step
planet_data = [
    ("Sun", 1.989e+30, 0.00e+00, 0.0, 0.0),
    ("Gamma-1", 2.612e+27, 8.67e+12, 32631.42, 44.44),
    ("Gamma-2", 4.304e+25, 2.07e+12, 4657.47, 20.18),
    ("Omega-3", 7.872e+21, 5.37e+12, 18968.09, 30.68),
    ("Alpha-4", 1.923e+20, 8.79e+12, 11066.83, 6.43),
    ("Omega-5", 1.310e+23, 3.75e+11, 23073.63, 42.77),
    ("Theta-6", 5.836e+21, 3.20e+12, 25350.04, 29.26),
    ("Theta-7", 4.347e+21, 3.49e+12, 10222.97, 23.86),
    ("Delta-8", 7.591e+27, 1.03e+13, 7952.18, 18.09),
]

N = len(planet_data)

m, p, v = np.empty(N), np.zeros((N,3)), np.zeros((N,3))
for i,(_, mass,a,spd,inc) in enumerate(planet_data):
    inc = radians(inc)
    m[i]=mass; p[i]=(a*cos(inc),a*sin(inc),0); v[i]=(-spd*sin(inc),spd*cos(inc),0)
v -= np.sum(m[:,None]*v,axis=0)/m.sum()
forces = np.zeros_like(p)

counts=np.full(size,N//size,int); counts[:N%size]+=1
start=np.insert(np.cumsum(counts)[:-1],0,0); seg=slice(start[rank],start[rank]+counts[rank])

def step():
    global p,v,forces
    v += 0.5*(forces/m[:,None])*DT; p += v*DT
    floc=np.zeros_like(forces)
    for i in range(seg.start,seg.stop):
        d=p-p[i]; d2=np.einsum("ij,ij->i",d,d); mask=d2>0
        inv=np.zeros_like(d2); inv[mask]=1/np.power(d2[mask],1.5)
        floc[i]=np.sum(d*(G*m[i]*m*inv)[:,None],axis=0)
    comm.Allreduce(floc,forces,op=MPI.SUM)
    v += 0.5*(forces/m[:,None])*DT

comm.Barrier()
start_time = MPI.Wtime()
warm_end   = start_time + WARMUP
bench_end  = warm_end + BENCH

# warm‑up
while MPI.Wtime() < warm_end:
    step()

# measurement: collect FPS every 0.1 s → O(secs/0.1) samples, small memory
sample_dt = 0.1  # seconds between FPS samples
steps, tmark = 0, MPI.Wtime()
local_fps = []
while MPI.Wtime() < bench_end:
    step(); steps += 1
    now = MPI.Wtime()
    if now - tmark >= sample_dt:
        fps = steps / (now - tmark)
        local_fps.append(fps)
        steps, tmark = 0, now

all_fps = comm.gather(np.array(local_fps,dtype=np.float64),root=0)
if rank==0:
    data=np.concatenate(all_fps) if all_fps else np.array([])
    data.sort(); n=data.size
    avg=data.mean() if n else 0
    low1=data[:max(1,int(n*0.01))].mean() if n else 0
    low0_1=data[:max(1,int(n*0.001))].mean() if n else 0
    print(f"\n=== MPI N‑body benchmark (aggregated) ===\nRuntime       : {BENCH}s + {WARMUP}s warmup\nRanks         : {size}\nSamples       : {n}\nAverage FPS   : {avg:10.1f}\n1 % Low FPS   : {low1:10.1f}\n0.1 % Low FPS : {low0_1:10.1f}\n", flush=True)

comm.Barrier()
