# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# File: entanglenodes.py
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

"""The EntangleNodes protocol creates entanglement between two remote nodes
linked by connection containing a quantum channel. One of the nodes (A) holds a source of entangled qubits.
Both nodes store their resulting qubit(s) on a quantum memory and schedule an
event once it is ready for retrieval.


.. aafig::
    :proportional:

    +---------------------+                                      +---------------------+
    |                     | +----------------------------------+ |                     |
    | "NodeA:"            | |                                  | | "NodeB:"            |
    | "QSource"           O-* "Connection: QuantumChannel -->" *-O "QuantumProcessor"  |
    | "QuantumProcessor"  | |                                  | |                     |
    |                     | +----------------------------------+ |                     |
    +---------------------+                                      +---------------------+

This network is constructed using the following function:

.. literalinclude:: ../../netsquid/examples/entanglenodes.py
    :pyobject: example_network_setup

If parts of the code above are a bit unclear, it may help to read `this part of the tutorial
<https://docs.netsquid.org/latest-release/tutorial.components.html#quantum-teleportation-using-components>`_.


The protocol that runs on both nodes A and B will be an :class:`~netsquid.examples.entanglenodes.EntangleNodes` protocol:

.. literalinclude:: ../../netsquid/examples/entanglenodes.py
    :pyobject: EntangleNodes
    :end-before: Parameters
    :append: ... <remaining code omitted>

The protocol will begin generating an entangled pair when a trigger event occurs,
and will backlog successive trigger calls so that a new pair will begin to be generated once the first pair is ready.
This can be seen in its `run()` method below where the event expression that starts a new round is yielded on:

.. literalinclude:: ../../netsquid/examples/entanglenodes.py
    :pyobject: EntangleNodes.run
    :dedent: 4

The following script runs an example simulation with a single round of entanglement generation
(i.e. initialising the :class:`~netsquid.examples.entanglenodes.EntangleNodes` protocol without a `start_expression`):

>>> import netsquid as ns
>>> print("This example module is located at: "
...       "{}".format(ns.examples.entanglenodes.__file__))
This example module is located at: .../netsquid/examples/entanglenodes.py
>>> from netsquid.examples.entanglenodes import example_network_setup, EntangleNodes
>>> network = example_network_setup()
>>> protocol_a = EntangleNodes(node=network.subcomponents["node_A"], role="source")
>>> protocol_b = EntangleNodes(node=network.subcomponents["node_B"], role="receiver")
>>> protocol_a.start()
>>> protocol_b.start()
>>> ns.sim_run()
>>> q1, = network.subcomponents["node_A"].qmemory.peek(0)
>>> q2, = network.subcomponents["node_B"].qmemory.peek(0)
>>> print("Fidelity of generated entanglement: {}".format(
...       ns.qubits.fidelity([q1, q2], ns.b00)))
Fidelity of generated entanglement: 0.99999...

"""
import netsquid as ns
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.protocols.protocol import Signals
from netsquid.components.instructions import INSTR_SWAP
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.util.simtools import sim_time
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel  # importing fibre delay model
from netsquid.components.models.qerrormodels import FibreLossModel # importing fibre loss model

from netsquid.components.qchannel import QuantumChannel
from netsquid.nodes.network import Network
from pydynaa import EventExpression
from netsquid.components.models import DepolarNoiseModel

__all__ = [
    "EntangleNodes",
    "example_network_setup",
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

    def __init__(self, node, role, start_expression=None, input_mem_pos=0, num_pairs=1, name=None):
        if role.lower() not in ["source", "receiver"]: # check if it is eighter source or receiver
            raise ValueError
            
        self._is_source = role.lower() == "source" # set the private boolean _is_source 
        
        name = name if name else "EntangleNode({}, {})".format(node.name, role) # check the name??
        
        super().__init__(node=node, name=name) # call the super consturctor
        
        if start_expression is not None and not isinstance(start_expression, EventExpression): # check the type of start_expression
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
            
        self.start_expression = start_expression
        self._num_pairs = num_pairs
        self._mem_positions = None
        
        # Claim input memory position:
        if self.node.qmemory is None:  # check if the memory is empty or not
            raise ValueError("Node {} does not have a quantum memory assigned.".format(self.node))
        self._input_mem_pos = input_mem_pos
        self._qmem_input_port = self.node.qmemory.ports["qin{}".format(self._input_mem_pos)]
        self.node.qmemory.mem_positions[self._input_mem_pos].in_use = True

    def start(self):
        self.entangled_pairs = 0  # counter
        self._mem_positions = [self._input_mem_pos]
        # Claim extra memory positions to use (if any):
        extra_memory = self._num_pairs - 1
        if extra_memory > 0:
            unused_positions = self.node.qmemory.unused_positions
            if extra_memory > len(unused_positions):
                raise RuntimeError("Not enough unused memory positions available: need {}, have {}"
                                   .format(self._num_pairs - 1, len(unused_positions)))
            for i in unused_positions[:extra_memory]:
                self._mem_positions.append(i)
                self.node.qmemory.mem_positions[i].in_use = True
        # Call parent start method
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
            for mem_pos in self._mem_positions[::-1]:
                # Iterate in reverse so that input_mem_pos is handled last
                if self._is_source:
                    self.node.subcomponents[self._qsource_name].trigger()
                yield self.await_port_input(self._qmem_input_port)
                if mem_pos != self._input_mem_pos:
                    self.node.qmemory.execute_instruction(
                        INSTR_SWAP, [self._input_mem_pos, mem_pos])
                    if self.node.qmemory.busy:
                        yield self.await_program(self.node.qmemory)
                self.entangled_pairs += 1
                self.send_signal(Signals.SUCCESS, mem_pos)

    @property
    def is_connected(self):
        if not super().is_connected:
            return False
        if self.node.qmemory is None:
            return False
        if self._mem_positions is None and len(self.node.qmemory.unused_positions) < self._num_pairs - 1:
            return False
        if self._mem_positions is not None and len(self._mem_positions) != self._num_pairs:
            return False
        if self._is_source:
            for name, subcomp in self.node.subcomponents.items():
                if isinstance(subcomp, QSource):
                    self._qsource_name = name
                    break
            else:
                return False
        return True


def example_network_setup(prep_delay=1e+3, num_mem_positions=20,channel_length = 0
                          ,memory_a_depolar_rate=1e+3,memory_b_depolar_rate=1e+3,channel_depolar_rate=1e+3):
    """Create an example network for use with the entangling nodes protocol.

    Parameters
    ----------
    prep_delay : float, optional
        Delay used in the source in this network. Default is 5 [ns].
    qchannel_delay : float, optional
        Delay of quantum channel. Default is 100 [ns].
    num_mem_positions : int
        Number of memory positions on both nodes in the network. Default is 3.

    Returns
    -------
    :class:`~netsquid.components.component.Component`
        A network component with nodes and channels as subcomponents.

    Notes
    -----
        This network is also used by the matching integration test.

    """
    # Setup nodes:
    depolar_rate=1e+6
    network = Network("Entangle_nodes")
    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    
    node_a.add_subcomponent(QuantumProcessor(  # add quanutm processor to the node a
        "QuantumMemoryATest", num_mem_positions, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(memory_a_depolar_rate)))
    
    node_b.add_subcomponent(QuantumProcessor( # add quanutm processor to the node b
        "QuantumMemoryBTest", num_mem_positions, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(memory_b_depolar_rate)))
    
    node_a.add_subcomponent(    # add bell state generator to the node a
        QSource("QSourceTest", state_sampler=StateSampler([ks.b00]),
                num_ports=2, status=SourceStatus.EXTERNAL,
                models={"emission_delay_model": FixedDelayModel(delay=prep_delay)}))
    
    # Creating delay model and photon loss model for quantum channel
    p_loss_init = 0.0
    p_loss_length  = 0.0
    
    delay_model = FibreDelayModel(c=200e3)
    loss_model = FibreLossModel(p_loss_init = p_loss_init, p_loss_length = p_loss_length)
    
    # Create and connect quantum channel:
    # defining the delay, photon loss, and noise models in channel.
    models={'delay_model':delay_model,'quantum_loss_model':loss_model, 'quantum_noise_model':DepolarNoiseModel(channel_depolar_rate)} 
    
    qchannel = QuantumChannel("QuantumChannelTest",length=channel_length, models = models)
    
    port_name_a, port_name_b = network.add_connection(
        node_a, node_b, channel_to=qchannel, label="quantum")
    
    # Setup Alice ports:
    node_a.subcomponents["QSourceTest"].ports["qout0"].forward_output(
        node_a.ports[port_name_a])
    node_a.subcomponents["QSourceTest"].ports["qout1"].connect(
        node_a.qmemory.ports["qin0"])
    # Setup Bob ports:
    node_b.ports[port_name_b].forward_input(node_b.qmemory.ports["qin0"])
    return network


if __name__ == "__main__":
    
    number_of_experiments = int(1e+4)
    fidelity_list = list();
    # distances = [1, 5, 10, 20, 50, 100 ,200] # distance vector, the unit is km.
    
    distances = [1, 2, 5, 20 , 30, 40 ,50]
    # distances = [30]
    times = []
    for distance in distances:
        fidelity = 0.0;
        
        for i in range(number_of_experiments):
            network = example_network_setup(channel_length = distance,memory_a_depolar_rate=1e+3,channel_depolar_rate=1e+3)
            protocol_a = EntangleNodes(node=network.get_node("node_A"), role="source")
            protocol_b = EntangleNodes(node=network.get_node("node_B"), role="receiver")
            protocol_a.start()
            protocol_b.start()
            time = sim_time()
            ns.sim_run()
            # print(sim_time() - time)
            q1, = network.get_node("node_A").qmemory.peek(0)
            # print(sim_time() - time)
            q2, = network.get_node("node_B").qmemory.peek(0)
            
            if q2 != None:
                fidelity = fidelity + ns.qubits.fidelity([q1, q2], ks.b00);
            # if i%100 == 0:
            # print(i)
        print(sim_time() - time)
        times.append(sim_time() - time)
        fidelity_list.append(fidelity/number_of_experiments)    
        print("For the distance equals to {} km the average fidelity is {}".format(distance,fidelity/number_of_experiments))

