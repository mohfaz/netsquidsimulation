#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 22:26:39 2021

@author: mohammad A processor with sync error 
"""
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.components.qprocessor import PhysicalInstruction
from netsquid.components.qprocessor import QuantumProcessor
import netsquid.components.instructions as instr
from netsquid.components.qmemory import QuantumMemory
import netsquid as ns
from netsquid.qubits import ketstates as ks

phys_instructions = [
    PhysicalInstruction(instr.INSTR_INIT, duration = 0,parallel=False),
    
    PhysicalInstruction(instr.INSTR_H, duration=0, parallel=False, topology=[0]),
    
    PhysicalInstruction(instr.INSTR_CNOT, duration=0, parallel=False,
                        topology=[(0,1)]),
        
    PhysicalInstruction(
        instr.INSTR_MEASURE, duration=0, parallel=False,

        apply_q_noise_after=False, topology=[3]),
    
    
    PhysicalInstruction(instr.INSTR_MEASURE, duration=0, parallel=True,
                        topology=[0, 1])
    
]

fidelity_list = [];




duration = [1,1e+3,2e+3,1e+4,1e+5,2e+5,1e+6,1e+9]
# duration = [2e+5]
for d in duration:
    print('for duration {}'.format(d))
    ph = PhysicalInstruction(instr.INSTR_X, duration=d, parallel=False, topology=[2, 3])    
    phys_instructions.append(ph)
    
    fidelity = 0
    
    noisy_qproc = QuantumProcessor("NoisyQPU", num_positions=4,
                                   mem_noise_models=[DepolarNoiseModel(1e3)] * 4,
                                   phys_instructions=phys_instructions)
    
    
    from netsquid.components.qprogram import QuantumProgram
    
    prog = QuantumProgram(num_qubits=4)
    q1, q2,q3,q4 = prog.get_qubit_indices(4)  # Get the qubit indices we'll be working with
    prog.apply(instr.INSTR_INIT, [q1, q2])
    prog.apply(instr.INSTR_H, q1)
    prog.apply(instr.INSTR_CNOT, [q1, q2])
    prog.apply(instr.INSTR_X,[q3])
    for _ in range(int(1e+4)):
        ns.sim_reset()
        
        noisy_qproc.reset()
        noisy_qproc.execute_program(prog, qubit_mapping = [0, 1,2,3])
        ns.sim_run()
        q1 = noisy_qproc.pop(0)
        q2 = noisy_qproc.pop(1)
        # print(q1,q2)
        fidelity = fidelity + ns.qubits.fidelity([q1[0], q2[0]], ks.b00);
    fidelity_list.append(fidelity/int(1e+4))