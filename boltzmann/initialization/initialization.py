import numpy as np


class Initialization:
    """Handles initialization instructions and creates
    PSV-Grid and p_flag.

    * Collects initialization :attr:`rules`
      and :attr:`instructions`, which apply these rules,
      in their respective list.
    * Categorizes each P-Grid point in :attr:`p_flag`,
      to specify its behavior in the simulation.
    * Creates a PSV-Grid (:attr:`psv`),
      with the instructed initial values.

    .. todo::
        - sphinx: link PSV-Grid to Calculation.data
          and p_flag to Calculation.p_flag
          in Initialization-Docstring
        - Figure out nice way to implement boundary points/p_flags
        - @Instruction: implement different 'shapes' to apply rules
          (e.g. a line with specified width,
          a ball with specified radius/diameter, ..).
        - @Instruction: current mode (cuboid)
          -> switch between convex hull and span?
        - Give Instruction.rule a reference to the rule,
          instead of the index (compare memory-use)

    Attributes
    ----------
    instructions : list
        Lists all instructions in the given order.
        Each element is an Instance of :class:`Instruction`
    rules : list
        Lists all defined rules.
        Each element is an Instance of :class:`InnerPointRule`.


    Notes
    -----
    Each P-Grid point fits
    into exactly one of the following Categories:

    * **Inner Point (Default)**:

      * both transport and collisions are applied normally

    * **Boundary Point**:

      * no collision step
      * additional reflection step after every transport step

    * **Constant Input/Output Point**:

      * no collision-step
      * no transport-step
      * Distribution is constant over the whole simulation.

    * **Time Variant Input/Output Point**:

      * no collision-step,
      * no transport-step,
      * Distribution is freshly initialized in every time step

    """
    SPECIFIED_CATEGORIES = ['inner_point',
                            'boundary_point',
                            'io_const',
                            'io_variant']

    def __init__(self):
        self.instructions = []
        self.rules = []
        self.p_flag = 0
        self.psv = 0

    def apply_rule(self,
                   rule,
                   p_min,
                   p_max):
        """Adds an Instruction, to apply the specified rule
        to the specified Are in P-Space

        Parameters
        ----------
        rule : int
            Index of the to be applied rule in
            :attr:`Initialization.rules`.
        p_min, p_max :  array_like(int)
            Indices of the boundaries on where to apply the rule.
            The rule is applied on all P-Grid points p, such that
            p_min[i] <= p[i] <= p_max[i]
            Indices are in vector form (indexing a non-flattened P-Grid).
            """
        instr = Instruction(rule, p_min, p_max)
        self.instructions.append(instr)
        return

    def add_rule_inner_points(self,
                              rho_list,
                              drift_list,
                              temp_list,
                              name=''):
        """Adds a new rule for initialization of inner points
        to the list of :attr:`rules`

        The added Rule initializes the velocity space of each specimen
        based on their conserved quantities
        mass (:attr:`rho`),
        mean velocity (:attr:`drift`)
        and temperature (:attr:`temp`).

        Parameters
        ----------
        rho_list : array_like
            List of the parameter rho, for each specimen.
            Rho correlates to the total weight/amount of particles in
            the area of the P-Grid point.,
        drift_list : array_like
            List of the parameter drift, for each specimen.
            Drift describes the mean velocity.
        temp_list : array_like
            List of the parameter temp, for each specimen.
            Temp describes the Temperature.
        name : str, optional
            Sets a name, for the points initialized with this rule.
        """
        new_rule = InnerPointRule(rho_list,
                                  drift_list,
                                  temp_list,
                                  name)
        self.rules.append(new_rule)
        return

    def run(self):
        print('')


class Instruction:
    """Encapsulates relevant information of initialization instruction

    The rule is applied on all P-Grid points p, such that
    p_min[i] <= p[i] <= p_max[i].

    Indices are in vector form (indexing a non-flattened P-Grid).

    Parameters
    ----------
    rule : int
        Rule-object, located in :attr:`Initialization.rules`.
    p_min, p_max :  array_like(int)
        Indices of the boundaries on where to apply the rule.
        The rule is applied on all P-Grid points p, such that
        p_min[i] <= p[i] <= p_max[i]
        Indices are in vector form (indexing a non-flattened P-Grid).

    Attributes
    ----------
    rule : int
    p_min, p_max : np.ndarray(int)
    """
    def __init__(self, rule, p_min, p_max):
        self.rule = rule
        self.p_min = np.array(p_min)
        self.p_max = np.array(p_max)
        assert self.p_min == np.minimum(p_min, p_max)
        assert self.p_max == np.maximum(p_min, p_max)
        return


class InnerPointRule:
    """Encapsulates all information necessary to initialize an inner point.

    Essentially initializes the velocity space of each specimen
    based on their conserved quantities
    mass (:attr:`rho`),
    mean velocity (:attr:`drift`)
    and temperature (:attr:`temp`).

    Parameters
    ----------
    rho_list : array_like
        List of the parameter rho, for each specimen.
        Rho correlates to the total weight/amount of particles in
        the area of the P-Grid point.,
    drift_list : array_like
        List of the parameter drift, for each specimen.
        Drift describes the mean velocity.
    temp_list : array_like
        List of the parameter temp, for each specimen.
        Temp describes the Temperature.
    name : str, optional
        Sets a name, for the points initialized with this rule.

    Attributes
    ----------
    cat : str
        Specifies the behavior in the calculation.
        :const:`Initialization.SPECIFIED_CATEGORIES`
        lists all possible categories.
    name : str
    rho = np.ndarray
    drift = np.ndarray
    temp = np.ndarray
    """
    def __init__(self,
                 rho_list,
                 drift_list,
                 temp_list,
                 name=''):
        assert len(rho_list) is len(drift_list)
        assert len(rho_list) is len(temp_list)
        assert all([len(drift) in [2, 3]
                    and len(drift) is len(drift_list[0])
                    for drift in drift_list])
        self.cat = 'inner_point'
        self.name = name
        self.rho = np.array(rho_list)
        self.drift = np.array(drift_list)
        self.temp = np.array(temp_list)
        return
