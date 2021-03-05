# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# File: purify.py
# 
# This file is part of the NetSquid package (https://netsquid.org).
# It is subject to the NetSquid Software End User License Conditions.
# A copy of these conditions can be found in the LICENSE.md file of this package.
# 
# NetSquid Authors
# ================
# 
# NetSquid is a being developed within the within the [quantum internet and networked computing roadmap](https://qutech.nl/roadmap/quantum-internet/) at QuTech. QuTech is a collaboration between TNO and the TUDelft.
# 
# Active authors (alphabetical):
# 
# - Tim Coopmans (scientific contributor)
# - Axel Dahlberg (scientific contributor)
# - Chris Elenbaas (software developer)
# - David Elkouss (scientific lead)
# - Rob Knegjens (software lead)
# - Martijn Papendrecht (software developer)
# - Ariana Torres Knoop (HPC contributor)
# - Stephanie Wehner (scientific lead)
# 
# Past authors (alphabetical):
# 
# - Damian Podareanu (HPC contributor)
# - Walter de Jong (HPC contributor)
# - Loek Nijsten (software developer)
# - Julio de Oliveira Filho (software architect)
# - Filip Rozpedek (scientific contributor)
# - Matt Skrzypczyk (software contributor)
# - Leon Wubben (software developer)
# 
# The simulation engine of NetSquid depends on the pyDynAA package,
# which is developed at TNO by Julio de Oliveira Filho, Rob Knegjens, Coen van Leeuwen, and Joost Adriaanse.
# 
# Ariana Torres Knoop, Walter de Jong and Damian Podareanu from SURFsara have contributed towards the optimization and parallelization of NetSquid.
# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This file uses NumPy style docstrings: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

"""Purification protocols are means to improve the fidelity of an already established entangled link.

The python file of this module contains example protocols for entanglement purification using filtering and distillation.
In the example below we will focus on filtering.

.. literalinclude:: ../../netsquid/examples/purify.py
    :pyobject: Filter
    :end-before: Parameters
    :append: ... <remaining code omitted>

The :class:`~netsquid.examples.purify.Filter` protocol and the :class:`~netsquid.examples.entanglenodes.EntangleNodes`
protocol are combined as sub-protocols in the `FilteringExample` protocol:

.. literalinclude:: ../../netsquid/examples/purify.py
    :pyobject: FilteringExample
    :end-before: Parameters
    :append: ... <remaining code omitted>

To be able to send classical messages back and forth between the nodes, we need to extend our network with a connection,
as can be seen in the :func:`~netsquid.examples.purify.example_network_setup()` function:


.. literalinclude:: ../../netsquid/examples/purify.py
    :pyobject: example_network_setup


Resulting in the following network:

.. aafig::
    :proportional:

    +---------------------+                                        +---------------------+
    |                     | +------------------------------------+ |                     |
    | "NodeA:"            | |                                    | | "NodeB:"            |
    | "QSource"           O-* "Connection: QuantumChannel -->"   *-O "QuantumProcessor"  |
    | "QuantumProcessor"  | |                                    | |                     |
    |                     | +------------------------------------+ |                     |
    |                     |                                        |                     |
    |                     | +------------------------------------+ |                     |
    |                     | |                                    | |                     |
    |                     O-* "Connection: ClassicalChannels <->"*-O                     |
    |                     | |                                    | |                     |
    |                     | +------------------------------------+ |                     |
    +---------------------+                                        +---------------------+


To collect data after each successful purification we add a :class:`~netsquid.util.datacollector.DataCollector` to our simulation:

.. literalinclude:: ../../netsquid/examples/purify.py
    :pyobject: example_sim_setup

Putting it all together you can run an example simulation like this:

>>> import netsquid as ns
>>> print("This example module is located at: {}".format(ns.examples.purify.__file__))
This example module is located at: .../netsquid/examples/purify.py
>>> from netsquid.examples.purify import example_network_setup, example_sim_setup
>>> network = example_network_setup()
>>> filt_example, dc = example_sim_setup(
...     network.get_node("node_A"), network.get_node("node_B"), num_runs=1000)
>>> filt_example.start()
>>> ns.sim_run()
>>> print("Average fidelity of generated entanglement with filtering: {}"
...       .format(dc.dataframe["F2"].mean()))
Average fidelity of generated entanglement with filtering: ...

.. W. Dur, H. J. Briegel, "Entanglement purification and quantum error
   correction", arXiv: 0705.4165

"""
import numpy as np
import netsquid as ns
import pydynaa as pd

from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1
from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, IGate, INSTR_SWAP
from netsquid.components.component import Message, Port
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qprogram import QuantumProgram
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models import DepolarNoiseModel
from netsquid.nodes.connections import DirectConnection
from pydynaa import EventExpression

__all__ = [
    "EntangleNodes",
    "Filter",
    "Distil",
    "FilteringExample",
    "example_network_setup",
    "example_sim_setup",
]

class EntangleNodes(NodeProtocol):
    """Cooperate with another node to generate shared entanglement.

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node`
        Node to run this protocol on.
    role : "source" or "receiver"
        Whether this protocol should act as a source or a receiver. Both are needed.
    start_expression : :class:`~pydynaa.core.EventExpression` or None, optional
        Event Expression to wait for before starting entanglement round.
    input_mem_pos : int, optional
        Index of quantum memory position to expect incoming qubits on. Default is 0.
    num_pairs : int, optional
        Number of entanglement pairs to create per round. If more than one, the extra qubits
        will be stored on available memory positions.
    name : str or None, optional
        Name of protocol. If None a default name is set.

    """

    def __init__(self, node, role, start_expression=None, input_mem_pos=0, num_pairs=3, name=None):
        if role.lower() not in ["source", "receiver"]:
            raise ValueError
        self._is_source = role.lower() == "source"
        name = name if name else "EntangleNode({}, {})".format(node.name, role)
        super().__init__(node=node, name=name)
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self.start_expression = start_expression
        self._num_pairs = num_pairs
        self._mem_positions = None
        # Claim input memory position:
        if self.node.qmemory is None:
            raise ValueError("Node {} does not have a quantum memory assigned.".format(self.node))
        self._input_mem_pos = input_mem_pos
        self._qmem_input_port = self.node.qmemory.ports["qin{}".format(self._input_mem_pos)]
        self.node.qmemory.mem_positions[self._input_mem_pos].in_use = True
        self.pro = None;
    def setProtocol_(self,protocol):
        self.pro = protocol;
        
    def start(self):
        self.entangled_pairs = 0  # counter
        # self._mem_positions = [self._input_mem_pos]
        # # Claim extra memory positions to use (if any):
        # extra_memory = self._num_pairs - 1
        # if extra_memory > 0:
        #     unused_positions = self.node.qmemory.unused_positions
        #     if extra_memory > len(unused_positions):
        #         raise RuntimeError("Not enough unused memory positions available: need {}, have {}"
        #                            .format(self._num_pairs - 1, len(unused_positions)))
        #     for i in unused_positions[:extra_memory]:
        #         self._mem_positions.append(i)
        #         self.node.qmemory.mem_positions[i].in_use = True
        # # Call parent start method
        self._mem_positions = list(range(2))
        return super().start()

    def stop(self):
        # Unclaim used memory positions:
        if self._mem_positions:
            for i in self._mem_positions[1:]:
                self.node.qmemory.mem_positions[i].in_use = False
            self._mem_positions = None
        # Call parent stop method
        super().stop()

    def run(self):
        while True:
            if self.start_expression is not None:
                yield self.start_expression
            elif self._is_source and self.entangled_pairs >= self._num_pairs:
                # If no start expression specified then limit generation to one round
                break
            while(True):
                mem_pos = 0;
                if self._is_source:
                        self.node.subcomponents[self._qsource_name].trigger()
                        
                yield self.await_port_input(self._qmem_input_port)
                if self.node.qmemory.mem_positions[1].is_empty:
                    mem_pos = 1
                    self.node.qmemory.execute_instruction(
                            INSTR_SWAP, [self._input_mem_pos, mem_pos])
                    if self.node.qmemory.busy:
                            yield self.await_program(self.node.qmemory)
                
                    
                self.entangled_pairs += 1
                self.send_signal(Signals.SUCCESS, mem_pos)
                if self._is_source:
                    yield self.await_signal(self.pro,Signals.READY)
                    print("wait for the ready signal from the source")
            # for mem_pos in self._mem_positions[::-1]:
            #         # Iterate in reverse so that input_mem_pos is handled last
            #     if not self.node.qmemory.mem_positions[mem_pos].is_empty:
            #         continue
                
            #     if self._is_source:
            #             self.node.subcomponents[self._qsource_name].trigger()
                        
            #     yield self.await_port_input(self._qmem_input_port)
                
            #     if mem_pos != self._input_mem_pos:
            #         self.node.qmemory.execute_instruction(
            #                 INSTR_SWAP, [self._input_mem_pos, mem_pos])
            #         if self.node.qmemory.busy:
            #                 yield self.await_program(self.node.qmemory)
                
            #     self.entangled_pairs += 1
            #     self.send_signal(Signals.SUCCESS, mem_pos)
            #     if self._is_source:
            #         yield self.await_signal(self.pro,Signals.READY)
            #         print("wait for the ready signal from the source")
    def _empty_position(self):
        for i in self._mem_positions:
            if self.node.qmemory.mem_positions[i].is_empty:
                return True
        return False
    @property
    def is_connected(self):
        if not super().is_connected:
            return False
        if self.node.qmemory is None:
            return False
        # if self._mem_positions is None and len(self.node.qmemory.unused_positions) < self._num_pairs - 1:
        #     return False
        # if self._mem_positions is not None and len(self._mem_positions) != self._num_pairs:
            # return False
        if self._is_source:
            for name, subcomp in self.node.subcomponents.items():
                if isinstance(subcomp, QSource):
                    self._qsource_name = name
                    break
            else:
                return False
        return True

class Filter(NodeProtocol):
    """Protocol that does local filtering on a node.

    This is done in combination with another node.

    Parameters
    ----------
    node : :py:class:`~netsquid.nodes.node.Node`
        Node with a quantum memory to run protocol on.
    port : :py:class:`~netsquid.components.component.Port`
        Port to use for classical IO communication to the other node.
    start_expression : :class:`~pydynaa.core.EventExpression` or None, optional
        Event expression node should wait for before starting filter.
        This event expression should have a
        :class:`~netsquid.protocols.protocol.Protocol` as source and should by fired
        by signalling a signal by this protocol, with the position of the qubit on the
        quantum memory as signal result.
        Must be set before the protocol can start
    msg_header : str, optional
        Value of header meta field used for classical communication.
    epsilon : float, optional
        Parameter used in filter's measurement operator.
    name : str or None, optional
        Name of protocol. If None a default name is set.

    Attributes
    ----------
    meas_ops : list
        Measurement operators to use for filter general measurement.

    """

    def __init__(self, node, port, start_expression=None, msg_header="filter",
                 epsilon=0.3, name=None):
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "Filter({}, {})".format(node.name, port.name)
        super().__init__(node, name)
        
        self.port = port
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression
        self.local_qcount = 0
        self.local_meas_OK = False
        self.remote_qcount = 0
        self.remote_meas_OK = False
        self.header = msg_header
        self._qmem_pos = None
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self._set_measurement_operators(epsilon)

    def _set_measurement_operators(self, epsilon):
        m0 = ops.Operator("M0", np.sqrt(epsilon) * outerprod(s0) + outerprod(s1))
        m1 = ops.Operator("M1", np.sqrt(1 - epsilon) * outerprod(s0))
        self.meas_ops = [m0, m1]

    def run(self):
        cchannel_ready = self.await_port_input(self.port)
        qmemory_ready = self.start_expression
        while True:
            # self.send_signal(Signals.WAITING)
            expr = yield cchannel_ready | qmemory_ready
            # self.send_signal(Signals.BUSY)
            if expr.first_term.value:
                classical_message = self.port.rx_input(header=self.header)
                if classical_message:
                    self.remote_qcount, self.remote_meas_OK = classical_message.items
                    self._handle_cchannel_rx()
            elif expr.second_term.value:
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                    event=expr.second_term.triggered_events[0], receiver=self)
                self._qmem_pos = ready_signal.result
                yield from self._handle_qubit_rx()

    # TODO does start reset vars?
    def start(self):
        self.local_qcount = 0
        self.remote_qcount = 0
        self.local_meas_OK = False
        self.remote_meas_OK = False
        return super().start()

    def stop(self):
        super().stop()
        # TODO should stop clear qmem_pos?
        if self._qmem_pos and self.node.qmemory.mem_positions[self._qmem_pos].in_use:
            self.node.qmemory.pop(positions=[self._qmem_pos])

    def _handle_qubit_rx(self):
        # Handle incoming Qubit on this node.
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        # Retrieve Qubit from input store
        output = self.node.qmemory.execute_instruction(INSTR_MEASURE, [self._qmem_pos], meas_operators=self.meas_ops)[0]
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        m = output["instr"][0]
        # m = INSTR_MEASURE(self.node.qmemory, [self._qmem_pos], meas_operators=self.meas_ops)[0]
        self.local_qcount += 1
        self.local_meas_OK = (m == 0)
        self.port.tx_output(Message([self.local_qcount, self.local_meas_OK], header=self.header))
        self._check_success()

    def _handle_cchannel_rx(self):
        # Handle incoming classical message from sister node.
        if (self.local_qcount == self.remote_qcount and
                self._qmem_pos is not None and
                self.node.qmemory.mem_positions[self._qmem_pos].in_use):
            self._check_success()

    def _check_success(self):
        # Check if protocol succeeded after receiving new input (qubit or classical information).
        # Returns true if protocol has succeeded on this node
        if (self.local_qcount > 0 and self.local_qcount == self.remote_qcount and
                self.local_meas_OK and self.remote_meas_OK):
            print("success")
            # SUCCESS!
            self.send_signal(Signals.SUCCESS, self._qmem_pos)
        elif self.local_meas_OK and self.local_qcount > self.remote_qcount:
            # Need to wait for latest remote status, i.e., remote does not send the meas ok yet.
            pass
        else:
            print("failure")
            # FAILURE
            self._handle_fail()
            self.send_signal(Signals.FAIL, self.local_qcount)

    def _handle_fail(self):
        if self.node.qmemory.mem_positions[self._qmem_pos].in_use:
            self.node.qmemory.pop(positions=[self._qmem_pos])

    @property
    def is_connected(self):
        if self.start_expression is None:
            return False
        if not self.check_assigned(self.port, Port):
            return False
        if not self.check_assigned(self.node, Node):
            return False
        if self.node.qmemory.num_positions < 1:
            return False
        return True


class Distil(NodeProtocol):
    """Protocol that does local DEJMPS distillation on a node.

    This is done in combination with another node.

    Parameters
    ----------
    node : :py:class:`~netsquid.nodes.node.Node`
        Node with a quantum memory to run protocol on.
    port : :py:class:`~netsquid.components.component.Port`
        Port to use for classical IO communication to the other node.
    role : "A" or "B"
        Distillation requires that one of the nodes ("B") conjugate its 
        rotation,
        while the other doesn't ("A").
    start_expression : :class:`~pydynaa.EventExpression`
        EventExpression node should wait for before starting distillation.
        The EventExpression should have a protocol as source, this protocol 
        should signal the quantum memory position
        of the qubit.
    msg_header : str, optional
        Value of header meta field used for classical communication.
    name : str or None, optional
        Name of protocol. If None a default name is set.
    k : number of consecutive purification that we do before conducting the results.
    """
    # set basis change operators for local DEJMPS step
    _INSTR_Rx = IGate("Rx_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0)))
    _INSTR_RxC = IGate("RxC_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0), conjugate=True))

    def __init__(self, node, port, role, start_expression=None, msg_header="distil", name=None, num_pos=2, number_of_purification_steps = 1, k = 1):
        if role.upper() not in ["A", "B"]: # check the roles
            raise ValueError
            
        conj_rotation = role.upper() == "B"
        
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port)) # check if the port is port
        
        name = name if name else "DistilNode({}, {})".format(node.name, port.name)
        
        super().__init__(node, name=name) # the super is nodeprotocol
        
        self.port = port
        
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression
        self._program = self._setup_dejmp_program(conj_rotation)
        
        # self.INSTR_ROT = self._INSTR_Rx if not conj_rotation else self._INSTR_RxC
        
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self.header = msg_header
        self._qmem_positions = [None]*num_pos
        
        self.num_pos = num_pos
        self._waiting_on_second_qubit = False
        
        # my codes
        self.first_time = True
        self.main_Qbit_pos = -1
        self.number_of_steps = number_of_purification_steps
        self.step = 0
        self.results = []
        self.start_time = sim_time()
        self.succeeded = False
        self.halt = False
        self.remote_control_qbit_pos = -1;
        self.k_step = 0
        self.k = k
        # queues
        
        self.queue_local_meas_results = []
        self.queue_local_control_qbit_mem_pos= []
        self.queue_remote_meas_results = []
        self.queue_remote_control_qbit_pos = []
        
        #
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))

    def _setup_dejmp_program(self, conj_rotation): # just setup the duetch et al purification protocol
        # this method is called by constructor
        INSTR_ROT = self._INSTR_Rx if not conj_rotation else self._INSTR_RxC
        prog = QuantumProgram(num_qubits=2)
        q1, q2 = prog.get_qubit_indices(2)
        prog.apply(INSTR_ROT, [q1])
        prog.apply(INSTR_ROT, [q2])
        prog.apply(INSTR_CNOT, [q1, q2])
        prog.apply(INSTR_MEASURE, q2, output_key="m", inplace=False)
        return prog
    

    def run(self):
        
       
        cchannel_ready = self.await_port_input(self.port) #receive the
        # classical message which includes the measurement results, number of q count and the header
        
        qmemory_ready = self.start_expression
        while True:
            # self.send_signal(Signals.WAITING)
            expr = yield cchannel_ready | qmemory_ready # I think this checks if channel ready or the memory is ready
            # self.send_signal(Signals.BUSY)
            if expr.first_term.value:
                classical_message = self.port.rx_input(header=self.header) # classical message
                classical_message_arrived = self.port.rx_input(header="entangle") # classical message
                if classical_message:
                    self.remote_control_qbit_pos, self.remote_meas_result = classical_message.items
                    print("The node {} receives a message from the other node: {}".format(self.name, self.remote_meas_result ))
                    self.queue_remote_control_qbit_pos.append(self.remote_control_qbit_pos)
                    self.queue_remote_meas_results.append(self.remote_meas_result)
                    self._check_success()
                if classical_message_arrived:
                    print(classical_message_arrived.items)
                    self.send_signal(Signals.READY)
                    #
            
            elif expr.second_term.value:
                
                source_protocol = expr.second_term.atomic_source
                
                ready_signal = source_protocol.get_signal_by_event(
                    event=expr.second_term.triggered_events[0], receiver=self)
                
                yield from self._handle_new_qubit(ready_signal.result)
 
    def start(self):
        # Clear any held qubits
        self._clear_qmem_positions()
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self._waiting_on_second_qubit = False
        ## My codes
        
        self.first_time = True
        self.main_Qbit_pos = -1
        
        
        ##
        return super().start()

    def _clear_qmem_positions(self):
        # invoked in start and _check_success
        # positions = [pos for pos in self._qmem_positions if pos is not None]
        for i in range(len(self.node.qmemory.mem_positions)):
            if not self.node.qmemory.mem_positions[i].is_empty:
                self.node.qmemory.pop(i)
        # self.results = []
        # if len(positions) > 0:
        #     self.node.qmemory.pop(positions=positions)
        self._qmem_positions = [None, None] # TODO what if the we use more than two qubits

    def _handle_new_qubit(self, memory_position): 
        # Process signalling of new entangled qubit
        # invoked buy run
        if self.name == "purify_B":
            self.port.tx_output( Message([self.name, "arrived"],
                                     header="entangle") )
        assert not self.node.qmemory.mem_positions[memory_position].is_empty # memory should not be empty
        if self._waiting_on_second_qubit:
            # Second qubit arrived: perform distil
            assert not self.node.qmemory.mem_positions[self._qmem_positions[0]].is_empty
            assert memory_position != self._qmem_positions[0]
            self._qmem_positions[1] = memory_position
            self._waiting_on_second_qubit = False
            yield from self._node_do_DEJMPS() # second qubit is hear. Do the dejmps purification
        else:
            # New candidate for first qubit arrived
            # Pop previous qubit if present:
            pop_positions = [p for p in self._qmem_positions if p is not None and p != memory_position]
            
            
            
            if self.main_Qbit_pos in pop_positions:
                pop_positions.remove(self.main_Qbit_pos)
                
            if len(pop_positions) > 0:
               pass # self.node.qmemory.pop(positions=pop_positions)
            # if memory_position == self.main_Qbit_pos:
            #     raise("khak bar saret")
            # Set new position:
            # self._qmem_positions[0] = memory_position # my codes
            
            # my codes
            if self.first_time:
                self._qmem_positions[0] = memory_position
                print("Node: ",self.name," Control Qbit is stored in ",memory_position,'\n time: ', sim_time() -  self.start_time)
                
                self.first_time = False
                self.main_Qbit_pos = memory_position
            
            self._qmem_positions[1] = None
            self.local_qcount += 1
            self.local_meas_result = None
            self._waiting_on_second_qubit = True # wait on the second qubit

    def _node_do_DEJMPS(self):
        # Perform DEJMPS distillation protocol locally on one node
        # Invoked by _handle_new_qubits method
        memory_positions = self._qmem_positions

        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        # We perform local DEJMPS
        yield self.node.qmemory.execute_program(self._program, [self.main_Qbit_pos, memory_positions[1]])  # If instruction not instant
        print("Node: ", self.name, ". The control Qbit {} is purified by the target Qbit {}".format(self.main_Qbit_pos, memory_positions[1]),'\n time: ', sim_time() -  self.start_time)
        self.local_meas_result = self._program.output["m"][0]
        print("The measurement result for the node {} and control qubit {} is {}".format(self.name, self.main_Qbit_pos ,self._program.output["m"][0]))
        
        self._qmem_positions[1] = None
        
        # Send local results to the remote node to allow it to check for success. #TODO optimistic approach: 
        ## My codes:
        self.step += 1
        self.results.append(self.local_meas_result)  
        if len(self.results) >= self.number_of_steps:
            self.port.tx_output( Message([self.main_Qbit_pos, self.results.copy()],
                                     header=self.header) )
            
            self.queue_local_meas_results.append(self.results.copy()) # the sequence of local measurements
            self.queue_local_control_qbit_mem_pos.append(self.main_Qbit_pos) # 
            
            self.results = [] 
            # self.main_Qbit_pos = -1  # reset main qubit pos index
            
            self._waiting_on_second_qubit = False # set the control qubit for the purification
            
            
            self.k_step += 1
            
            self._check_success() # check if the purification result is successful or not
            

        
        # self.port.tx_output(Message([self.local_qcount, self.local_meas_result],
                                    # header=self.header))

    def _check_success(self):
        # Check if distillation succeeded by comparing local and remote results
        # used by run() method
        
        # if self.name == "purify_A":
        #     print("yes",'\n time: ', sim_time() -  self.start_time)
        found = False
        if ( self.k_step >= self.k and
               len(self.queue_local_meas_results) > 0 and
                len(self.queue_remote_control_qbit_pos) > 0):
            
            # a1 = self.queue_local_meas_results.pop(0) # assuming the classical channel in perfect without loss also the the quantum channel is  lost
            for i in range(min(len(self.queue_local_control_qbit_mem_pos),len(self.queue_remote_control_qbit_pos))):
                if self.queue_local_control_qbit_mem_pos[i] == self.queue_remote_control_qbit_pos[i]:
                    p_local = self.queue_local_control_qbit_mem_pos.pop(i)
                    p_remote = self.queue_remote_control_qbit_pos.pop(i)
                    meas_remote = self.queue_remote_meas_results.pop(i)
                    meas_local = self.queue_local_meas_results.pop(i)
                    found = True
                    break
            if found and meas_remote  == meas_local:
                # SUCCESS
                self.send_signal(Signals.SUCCESS, p_local)
                print("Node ",self.name," success",'\ntime: ', sim_time() -  self.start_time)
                self.succeeded = True
                
            else:
                print("Node ",self.name," FAILURE",'\ntime: ', sim_time() -  self.start_time)
                # FAILURE
                self.node.qmemory.pop(0)
                self.node.qmemory.pop(1)
                self.send_signal(Signals.FAIL, self.local_qcount)
                self.first_time = True
                self.main_Qbit_pos = -1
            self.results = []    
            self.local_meas_result = None
            self.remote_meas_result = None
            self._qmem_positions = [None, None]
            self._waiting_on_second_qubit = False
            self.first_time = True
            self.main_Qbit_pos = -1
            self.k_step = 0
            # self._clear_qmem_positions()
            
    @property
    def is_connected(self):
        if self.start_expression is None:
            return False
        if not self.check_assigned(self.port, Port):
            return False
        if not self.check_assigned(self.node, Node):
            return False
        if self.node.qmemory.num_positions < 2:
            return False
        return True
class JustEntangle(LocalProtocol):
    r"""Protocol for a complete filtering experiment.

    Combines the sub-protocols:
    - :py:class:`~netsquid.examples.entanglenodes.EntangleNodes`
    - :py:class:`~netsquid.examples.purify.Filter`

    Will run for specified number of times then stop, recording results after each run.

    Parameters
    ----------
    node_a : :py:class:`~netsquid.nodes.node.Node`
        Must be specified before protocol can start.
    node_b : :py:class:`~netsquid.nodes.node.Node`
        Must be specified before protocol can start.
    num_runs : int
        Number of successful runs to do.
    epsilon : float
        Parameter used in filter's measurement operator.

    Attributes
    ----------
    results : :py:obj:`dict`
        Dictionary containing results. Results are :py:class:`numpy.array`\s.
        Results keys are *F2*, *pairs*, and *time*.

    Subprotocols
    ------------
    entangle_A : :class:`~netsquid.examples.entanglenodes.EntangleNodes`
        Entanglement generation protocol running on node A.
    entangle_B : :class:`~netsquid.examples.entanglenodes.EntangleNodes`
        Entanglement generation protocol running on node B.
    purify_A : :class:`~netsquid.examples.purify.Filter`
        Purification protocol running on node A.
    purify_B : :class:`~netsquid.examples.purify.Filter`
        Purification protocol running on node B.

    Notes
    -----
        The filter purification does not support the stabilizer formalism.

    """

    def __init__(self, node_a, node_b, num_runs, num_pos = 2):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="Distilling example")
        
        self.num_runs = num_runs
        # Initialise sub-protocols
        
        self.add_subprotocol(EntangleNodes(node=node_a, role="source", input_mem_pos=0,
                                           num_pairs=num_pos, name="entangle_A")) # node a adding entangle nodes
        
        self.add_subprotocol(
            EntangleNodes(node=node_b, role="receiver", input_mem_pos=0, num_pairs=num_pos,
                          name="entangle_B")) # node b adding entangle nodes.
       
        # self.add_subprotocol(Filter(node_a, node_a.get_conn_port(node_b.ID), #TODO add distilation instructor
        #                             epsilon=epsilon, name="purify_A"))
        # self.add_subprotocol(Filter(node_b, node_b.get_conn_port(node_a.ID), #TODO add distilaiton instructor
        #                             epsilon=epsilon, name="purify_B"))
        self.add_subprotocol(Distil(node= node_a,
        port= node_a.get_conn_port(node_b.ID), role='A',name = "purify_A", num_pos = num_pos))
       
        self.add_subprotocol(Distil(node= node_b,
        port= node_b.get_conn_port(node_a.ID), role='B',name = "purify_B", num_pos = num_pos))
        # Set start expressions
        self.subprotocols["purify_A"].start_expression = (
            self.subprotocols["purify_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_B"].start_expression = (
            self.subprotocols["purify_B"].await_signal(self.subprotocols["entangle_B"],
                                                       Signals.SUCCESS))
        start_expr_ent_A = (self.subprotocols["entangle_A"].await_signal(
                            self.subprotocols["purify_A"], Signals.FAIL) |
                            self.subprotocols["entangle_A"].await_signal(
                                self, Signals.WAITING))
        self.subprotocols["entangle_A"].start_expression = start_expr_ent_A
        
    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            
            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["purify_A"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["purify_B"], Signals.SUCCESS))
            signal_A = self.subprotocols["purify_A"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            signal_B = self.subprotocols["purify_B"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            result = {
                "pos_A": signal_A,
                "pos_B": signal_B,
                "time": sim_time() - start_time,
                "pairs": self.subprotocols["entangle_A"].entangled_pairs,
            }
            self.send_signal(Signals.SUCCESS, result)
            ns.sim_reset()

class DistilExample(LocalProtocol):
    r"""Protocol for a complete filtering experiment.

    Combines the sub-protocols:
    - :py:class:`~netsquid.examples.entanglenodes.EntangleNodes`
    - :py:class:`~netsquid.examples.purify.Filter`

    Will run for specified number of times then stop, recording results after each run.

    Parameters
    ----------
    node_a : :py:class:`~netsquid.nodes.node.Node`
        Must be specified before protocol can start.
    node_b : :py:class:`~netsquid.nodes.node.Node`
        Must be specified before protocol can start.
    num_runs : int
        Number of successful runs to do.
    epsilon : float
        Parameter used in filter's measurement operator.
    k : the batch size that we want to purify consecutively 
    before conducting the results
    Attributes
    ----------
    results : :py:obj:`dict`
        Dictionary containing results. Results are :py:class:`numpy.array`\s.
        Results keys are *F2*, *pairs*, and *time*.

    Subprotocols
    ------------
    entangle_A : :class:`~netsquid.examples.entanglenodes.EntangleNodes`
        Entanglement generation protocol running on node A.
    entangle_B : :class:`~netsquid.examples.entanglenodes.EntangleNodes`
        Entanglement generation protocol running on node B.
    purify_A : :class:`~netsquid.examples.purify.Filter`
        Purification protocol running on node A.
    purify_B : :class:`~netsquid.examples.purify.Filter`
        Purification protocol running on node B.

    Notes
    -----
        The filter purification does not support the stabilizer formalism.

    """

    def __init__(self, node_a, node_b, num_runs, num_pos = 2,number_of_purification_steps =1, k = 1):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="Distilling example")
        
        self.num_runs = num_runs
        # Initialise sub-protocols
        
        # Entangelment  Protocols
        self.add_subprotocol( EntangleNodes(node=node_a, role="source", input_mem_pos=0,
                                           num_pairs=2*number_of_purification_steps, name="entangle_A") )  # node a adding entangle nodes
        self.add_subprotocol(
            EntangleNodes(node=node_b, role="receiver", input_mem_pos=0, num_pairs=2*number_of_purification_steps,
                          name="entangle_B") 
            ) # node b adding entangle nodes.
       
        # Distilation Protocols
        self.add_subprotocol(Distil(node= node_a,
        port= node_a.get_conn_port(node_b.ID), role='A',name = "purify_A",
        num_pos = num_pos,number_of_purification_steps=number_of_purification_steps, k = k))
       
        self.add_subprotocol(Distil(node= node_b,
        port= node_b.get_conn_port(node_a.ID), role='B',name = "purify_B",
        num_pos = num_pos,number_of_purification_steps=number_of_purification_steps, k = k))
       
        self.subprotocols["entangle_A"].setProtocol_(self.subprotocols["purify_A"])

        # Set start expressions
        self.subprotocols["purify_A"].start_expression = (
            self.subprotocols["purify_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.SUCCESS))
        
        self.subprotocols["purify_B"].start_expression = (
            self.subprotocols["purify_B"].await_signal(self.subprotocols["entangle_B"],
                                                       Signals.SUCCESS))
        start_expr_ent_A = (
                            self.subprotocols["entangle_A"].await_signal(
                            self.subprotocols["purify_A"], Signals.FAIL) |
                            self.subprotocols["entangle_A"].await_signal(
                                self, Signals.WAITING) )
        self.subprotocols["entangle_A"].start_expression = start_expr_ent_A
        
    def run(self):
        node_b = []
        node_a = []
        node_a_temp = []
        node_b_temp = []
        
        self.start_subprotocols()
        for i in range(self.num_runs): # Note the number of runs here
            print("The run number  is: ",i)
            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)
            
            # while len(node_a)== 0 or len(node_b)==0:
                
            #     yield (self.await_signal(self.subprotocols["purify_B"], Signals.SUCCESS) | self.await_signal(self.subprotocols["purify_A"], Signals.SUCCESS))
            # # self.node_b.append(self.subprotocols["purify_B"].get_signal_result(Signals.SUCCESS,
            # #                                                            self))
                
            #     signal_A = self.subprotocols["purify_A"].get_signal_result(Signals.SUCCESS,
            #                                                            self)
            #     signal_B = self.subprotocols["purify_B"].get_signal_result(Signals.SUCCESS,
            #                                                            self)
                
            #     if signal_A != None and not signal_A in node_a_temp:
            #         node_a.append(signal_A)
            #         node_a_temp.append(signal_A)                    
            #     if signal_B != None and not signal_B in node_b_temp:
            #         node_b.append(signal_B)
            #         node_b_temp.append(signal_B)
                    
            # if (len(node_a_temp) == 20):
            #     node_a_temp = []
            # if (len(node_b_temp) == 20):
            #     node_b_temp = []
            yield (self.await_signal(self.subprotocols["purify_B"], Signals.SUCCESS) & self.await_signal(self.subprotocols["purify_A"], Signals.SUCCESS))
            signal_A = self.subprotocols["purify_A"].get_signal_result(Signals.SUCCESS,
                                                                        self)
            signal_B = self.subprotocols["purify_B"].get_signal_result(Signals.SUCCESS,
                                                                        self)
            result = {
                "pos_A": signal_A,
                "pos_B": signal_B,
                "time": sim_time() - start_time,
                "pairs": self.subprotocols["entangle_A"].entangled_pairs,
            }
            self.send_signal(Signals.SUCCESS, result)
            print("///////////////////////////////////////")
            # self.reset()
            # Maybe it is a good idea to send the failure sigal too.
            

class FilteringExample(LocalProtocol):
    r"""Protocol for a complete filtering experiment.

    Combines the sub-protocols:
    - :py:class:`~netsquid.examples.entanglenodes.EntangleNodes`
    - :py:class:`~netsquid.examples.purify.Filter`

    Will run for specified number of times then stop, recording results after each run.

    Parameters
    ----------
    node_a : :py:class:`~netsquid.nodes.node.Node`
        Must be specified before protocol can start.
    node_b : :py:class:`~netsquid.nodes.node.Node`
        Must be specified before protocol can start.
    num_runs : int
        Number of successful runs to do.
    epsilon : float
        Parameter used in filter's measurement operator.

    Attributes
    ----------
    results : :py:obj:`dict`
        Dictionary containing results. Results are :py:class:`numpy.array`\s.
        Results keys are *F2*, *pairs*, and *time*.

    Subprotocols
    ------------
    entangle_A : :class:`~netsquid.examples.entanglenodes.EntangleNodes`
        Entanglement generation protocol running on node A.
    entangle_B : :class:`~netsquid.examples.entanglenodes.EntangleNodes`
        Entanglement generation protocol running on node B.
    purify_A : :class:`~netsquid.examples.purify.Filter`
        Purification protocol running on node A.
    purify_B : :class:`~netsquid.examples.purify.Filter`
        Purification protocol running on node B.

    Notes
    -----
        The filter purification does not support the stabilizer formalism.

    """

    def __init__(self, node_a, node_b, num_runs, epsilon=0.3):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="Filtering example")
        
        self._epsilon = epsilon
        self.num_runs = num_runs
        # Initialise sub-protocols
        
        self.add_subprotocol(EntangleNodes(node=node_a, role="source", input_mem_pos=0,
                                           num_pairs=1, name="entangle_A"))
        self.add_subprotocol(
            EntangleNodes(node=node_b, role="receiver", input_mem_pos=0, num_pairs=1,
                          name="entangle_B"))
       
        self.add_subprotocol(Filter(node_a, node_a.get_conn_port(node_b.ID),
                                    epsilon=epsilon, name="purify_A"))
        self.add_subprotocol(Filter(node_b, node_b.get_conn_port(node_a.ID),
                                    epsilon=epsilon, name="purify_B"))
        
        
        # Set start expressions
        self.subprotocols["purify_A"].start_expression = (
            self.subprotocols["purify_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_B"].start_expression = (
            self.subprotocols["purify_B"].await_signal(self.subprotocols["entangle_B"],
                                                       Signals.SUCCESS))
        start_expr_ent_A = (self.subprotocols["entangle_A"].await_signal(
                            self.subprotocols["purify_A"], Signals.FAIL) |
                            self.subprotocols["entangle_A"].await_signal(
                                self, Signals.WAITING))
        self.subprotocols["entangle_A"].start_expression = start_expr_ent_A

    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            print("The run number: ", i)
            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["purify_A"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["purify_B"], Signals.SUCCESS))
            signal_A = self.subprotocols["purify_A"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            signal_B = self.subprotocols["purify_B"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            result = {
                "pos_A": signal_A,
                "pos_B": signal_B,
                "time": sim_time() - start_time,
                "pairs": self.subprotocols["entangle_A"].entangled_pairs,
            }
            self.send_signal(Signals.SUCCESS, result)
            print("////////////////////////////////////////////////////")


def example_network_setup(source_delay=3, source_fidelity_sq=1, depolar_rate=1e+3,
                          node_distance=20, purification_steps = 1):
    """Create an example network for use with the purification protocols.

    Returns
    -------
    :class:`~netsquid.components.component.Component`
        A network component with nodes and channels as subcomponents.

    Notes
    -----
        This network is also used by the matching integration test.

    """
    # num_positions = max(int(2*node_distance/(source_delay*source_delay/1e+9)),2)
    # num_positions = min(num_positions,20)
    # print(num_positions)
    num_positions = 2
    network = Network("purify_network")

    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    
    node_a.add_subcomponent(QuantumProcessor(
        "QuantumMemory_A", num_positions=num_positions, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(depolar_rate)))
    
    state_sampler = StateSampler(
        [ks.b01, ks.s00],
        probabilities=[source_fidelity_sq, 1 - source_fidelity_sq])
    
    node_a.add_subcomponent(QSource(
        "QSource_A", state_sampler=state_sampler,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=2, status=SourceStatus.EXTERNAL))
    
    node_b.add_subcomponent(QuantumProcessor(
        "QuantumMemory_B", num_positions=num_positions, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(depolar_rate)))
    
    conn_cchannel = DirectConnection(
        "CChannelConn_AB",
        ClassicalChannel("CChannel_A->B", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_B->A", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}))
    
    network.add_connection(node_a, node_b, connection=conn_cchannel)
    # node_A.connect_to(node_B, conn_cchannel)
    
    qchannel = QuantumChannel("QChannel_A->B", length=node_distance,
                              models={"quantum_loss_model": None,
                                      "delay_model": FibreDelayModel(c=200e3)},
                              depolar_rate=depolar_rate)
    
    port_name_a, port_name_b = network.add_connection(
        node_a, node_b, channel_to=qchannel, label="quantum")
    
    # Link Alice ports:
    node_a.subcomponents["QSource_A"].ports["qout1"].forward_output(
        node_a.ports[port_name_a])
    
    node_a.subcomponents["QSource_A"].ports["qout0"].connect(
        node_a.qmemory.ports["qin0"])
    
    # Link Bob ports:
    node_b.ports[port_name_b].forward_input(node_b.qmemory.ports["qin0"])
    
    return network


def example_sim_setup(node_a, node_b, num_runs, epsilon=0.3, purify = "filter", number_of_purification_steps = 1,k = 1):
    """Example simulation setup for purification protocols.

    Returns
    -------
    :class:`~netsquid.examples.purify.FilteringExample`
        Example protocol to run.
    :class:`pandas.DataFrame`
        Dataframe of collected data.

    """
    if purify == "filter":
        filt_example = FilteringExample(node_a, node_b, num_runs=num_runs, epsilon=epsilon)
    elif purify == "distil":
        distil_example = DistilExample(node_a, node_b, num_runs = num_runs,
        num_pos = node_a.qmemory.num_positions,
        number_of_purification_steps =number_of_purification_steps,k = k)
        
    elif purify == "None":
        pass
    else:
        raise("variable error for the distilation/filtering methodology")
        
    def record_run(evexpr):
        # Callback that collects data each run
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        # Record fidelity
        q_A, = node_a.qmemory.pop(positions=[result["pos_A"]])
        q_B, = node_b.qmemory.pop(positions=[result["pos_B"]])
        print(result["pos_A"])
        print(result["pos_B"])
        f2 = qapi.fidelity([q_A, q_B], ks.b01, squared=True)
        return {"F2": f2, "pairs": result["pairs"], "time": result["time"]}
    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    if purify == "filter":
        dc.collect_on(pd.EventExpression(source=filt_example,
                                         event_type=Signals.SUCCESS.value))
        return filt_example, dc
    
    else:
        dc.collect_on(pd.EventExpression(source=distil_example,
                                             event_type=Signals.SUCCESS.value))
        return distil_example, dc
            
class logger():
    def __init__( self,writing = True):
        self.writing = writing
    def logout(self,str):
        if self.writing:
            print(str)

if __name__ == "__main__":
    num_runs = int(1)
    fidelity_list = []
    times = []
    distances = [1,2,5,20,30,40,50]
    number_of_purification_step  = 1 # number of consecutive 
    k = 3 # maximum number of purification step
    # distances = [30]
    # distances  = [1000]
    # distances  = [200e+3]
    distances = [1]
    for distance in distances:
        ns.sim_reset()
        print("The distance is ", distance, " km")
        network = example_network_setup(node_distance = distance,purification_steps = number_of_purification_step)
        filt_example, dc = example_sim_setup(network.get_node("node_A"),
                                         network.get_node("node_B"),
                                         num_runs=num_runs, purify = "distil",number_of_purification_steps = number_of_purification_step, k = k )
        filt_example.start()
        ns.sim_run()
        
        fidelity = dc.dataframe["F2"].mean()
        times.append(dc.dataframe["time"].mean())
        print( "Average fidelity of generated entanglement with distilation: {}".format(fidelity) )
        fidelity_list.append(fidelity)
        print(times)
        ns.sim_stop()
        