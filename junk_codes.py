
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
