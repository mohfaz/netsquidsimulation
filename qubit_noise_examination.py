#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:02:34 2021

@author: mohammad
"""
# fidelity  = 0
# for _ in range(int(1e+4)):
    
#     q1, q2 = ns.qubits.create_qubits(2)
#     prob =  0.181269246922018;
        
#     ns.qubits.operate(q1, ns.H)
#     ns.qubits.operate([q1,q2], ns.CNOT)
#     # ns.qubits.combine_qubits([q1, q2])
#     depolarize(q1, prob)
#     depolarize(q2, prob)
#     # ns.qubits.operate(q1, ns.Z)
#     # ns.qubits.operate(q2, ns.Z)
    
#     fidelity = fidelity + ns.qubits.fidelity([q1, q2], ks.b00);

# print(fidelity/1e+4)


import netsquid as ns
from netsquid.qubits.qubitapi import depolarize
from netsquid.qubits import ketstates as ks
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.components import QuantumMemory
from netsquid.qubits.qubitapi import create_qubits
import netsquid.qubits.operators as ops

depolar_noise = DepolarNoiseModel(depolar_rate=1e3)

fidelity = 0.0

for _ in range(int(1e+4)):
    ns.sim_reset()
    qmem = QuantumMemory(name="MyMemory", num_positions=2,memory_noise_models=[depolar_noise, depolar_noise])
    qubits = create_qubits(2)
    qmem.put(qubits)
    qmem.operate(ops.H,positions=[0])
    qmem.operate(ops.CNOT,positions=[0,1])
    # print(qmem.delta_time( positions=[0,1]))
    ns.sim_run(2e+5)
    
    # print(ns.sim_time())
    q1 = qmem.peek(0)
    q2 = qmem.peek(1)
    # print(qmem.delta_time( positions=[0,1]))
    
    fidelity = fidelity + ns.qubits.fidelity([q1[0], q2[0]], ks.b00)
    
print(fidelity / 1e+4)