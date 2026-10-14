import numpy as np
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------------
# Fixed framework
# ----------------------------------------------------------------------------
TOTAL_HOURS = 120
BINS_PER_HOUR = 3                         # 1 bin = 20 min
N_TIME_BINS = TOTAL_HOURS * BINS_PER_HOUR  # 360
CELL_CYCLE_HOURS = 12

LATTICE_SIZE = 18                          # 18**2 + 2 * 17**2 = 902 fibers
N_TYPE0 = LATTICE_SIZE ** 2                # 324

# Tangential migration is allowed only while a neuron is at bins 72-108 after birth (24-36 h).
IMZ_START = 72
IMZ_END = 108

GENERATION_DURATION_BINS = 101             # half-normal SCPN generation probability (~34 h, Fig. 3d)

# Exploratory parameters (Fig. 3c; 4 * 4 * 3 * 4 * 4 * 3 = 2,304 combinations)
PARAMETER_GRID = {
    "cycle_diff": [0.5, 0.3, 0.1, 0.01],                         # (1) phase difference
    "cyclerandomness": [0.01, 0.1, 0.3, 0.5],                    # (2) phase variation (deviation)
    "IPratio": [0.1, 0.3, 0.5],                                  # (3) generation ratio of IPs
    "tangentialmigrationprob": [1 / 36, 1 / 12, 1 / 6, 0.5],     # (4) frequency of tangential migration
    "tangentialmigrationcellcycle": [0.2, 0.5, 0.8, 1.0],        # (5) phase specificity of migration
    "trapeffectiveheight": [0, 36, 360],                         # (6) axon trapping
}
# trapeffectiveheight (bins) -> trapping starts (IMZ_END + value) / 3 h after generation:
# 0 -> 36 h, 36 -> 48 h, 360 -> beyond the simulated period (control without trapping).


def build_cell_generation_prob(duration_bins=GENERATION_DURATION_BINS, n_time_bins=N_TIME_BINS):
    """Half-normal SCPN generation probability: 1 at t = 0, decaying to 0 after duration_bins (Fig. 3d)."""
    duration_bins = max(1, min(int(duration_bins), n_time_bins))
    r = np.linspace(0, 3, duration_bins)
    pdf1 = np.exp(-r ** 2 / 2)
    pdf1 = pdf1 - pdf1.min()
    pdf1 = pdf1 / pdf1.max()
    return np.concatenate([pdf1, np.zeros(n_time_bins - len(pdf1))])


CELL_GENERATION_PROB = build_cell_generation_prob()


def resolve_seed(seed=None):
    """Return seed, or a random base seed in [0, 2**31) if seed is None.

    The base seed is printed so that a run with a random seed can be reproduced with --seed.
    Offsets added to it by the scripts stay below 2**32, the limit of np.random.seed.
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2 ** 31)
    print(f"base seed: {seed}")
    return seed


def round_cell_cycle(cell_cycle):
    """Wrap a cell cycle state into [0, 1)."""
    return cell_cycle - np.floor(cell_cycle)


class RadialFiber:
    """One RGC and its radial fiber."""

    def __init__(self, id, position, cellcycle):
        self.id = id
        self.position = position
        self.cellcycle = cellcycle
        self.adjacent_radialfibers = None
        self.trap = False

    def pass_time(self, cell_cycle_hours, bins_per_hour):
        self.cellcycle += 1 / cell_cycle_hours / bins_per_hour
        self.cellcycle = self.cellcycle - np.floor(self.cellcycle)


class Neuron:
    """A newborn neuron: the fiber it is attached to and its height (bins since birth)."""

    def __init__(self, radial_fiber, position):
        self.radial_fiber = radial_fiber
        self.position = position


def generate_lattice_positions(size=LATTICE_SIZE):
    """Return fiber coordinates (unit = distance between adjacent type-0 fibers) and types (0/1)."""
    positions = []
    types = []
    for i in range(size):
        for j in range(size):
            positions.append([i * 3 ** 0.5 / 2.0, j - i / 2.0])
            types.append(0)
    for i in range(size - 1):
        for j in range(size - 1):
            positions.append([i * 3 ** 0.5 / 2.0 + 3 ** 0.5 / 3.0, j - i / 2.0])
            types.append(1)
    for i in range(size - 1):
        for j in range(size - 1):
            positions.append([i * 3 ** 0.5 / 2.0 + 3 ** 0.5 / 6.0, j - i / 2.0 + 0.5])
            types.append(1)
    return np.array(positions), np.array(types)


def build_adjacency(radial_fiber_list):
    """Fibers within 1.1 x the nearest-neighbor distance are neighbors (6 per fiber except at the border)."""
    positions = np.array([rf.position for rf in radial_fiber_list])
    tree = cKDTree(positions)
    dist, _ = tree.query(positions, k=2)
    min_distance = np.min(dist[:, 1])
    adjacency = [[] for _ in radial_fiber_list]
    for a, b in tree.query_pairs(r=1.1 * min_distance):
        adjacency[a].append(b)
        adjacency[b].append(a)
    for rf, neighbor_idx in zip(radial_fiber_list, adjacency):
        rf.adjacent_radialfibers = [radial_fiber_list[k] for k in neighbor_idx]


def define_radial_fibers(cycle_randomness, cycle_diff, disorder_level=0.0, rng=None, size=LATTICE_SIZE):
    """Place the fibers on the hexagonal lattice and set their initial cell cycle states.

    Phase "0" fibers start at 1 - 2*cycle_randomness - cycle_randomness*U(0, 1) and phase "pi" fibers
    start cycle_diff earlier. With disorder_level = 0 (original simulation), type-0 fibers are phase "0"
    and type-1 fibers are phase "pi". With disorder_level = d (Extended Data Fig. 6e, f), each fiber is
    independently reassigned with probability d to phase "0" or "pi" with equal probability.
    """
    if rng is None:
        rng = np.random.default_rng()
    positions, types = generate_lattice_positions(size)
    n_total = len(positions)
    canonical_is_phase0 = (types == 0)
    reshuffle = rng.random(n_total) < disorder_level
    random_assignment = rng.random(n_total) < 0.5
    is_phase0 = np.where(reshuffle, random_assignment, canonical_is_phase0)

    radial_fiber_list = []
    for idx, (pos, phase0) in enumerate(zip(positions, is_phase0)):
        if phase0:
            cellcycle = round_cell_cycle(1 - cycle_randomness * 2 - cycle_randomness * np.random.rand())
        else:
            cellcycle = round_cell_cycle(1 - cycle_randomness * 2 - cycle_diff - cycle_randomness * np.random.rand())
        rf = RadialFiber(idx, list(pos), cellcycle)
        rf.type = int(types[idx])
        rf.phase_group = 0 if phase0 else 1
        radial_fiber_list.append(rf)
    build_adjacency(radial_fiber_list)
    return radial_fiber_list


def simulation(radial_fiber_list, trap_effective_height, tangential_migration_cell_cycle,
               tangential_migration_prob, IP_ratio, cell_cycle_hours=CELL_CYCLE_HOURS,
               cell_generation_prob=CELL_GENERATION_PROB, record_cell_cycle=False):
    """Run the simulation for N_TIME_BINS bins.

    Each time bin:
      (1) fibers carrying a neuron at height >= trap_effective_height + IMZ_END become trapped
      (2) tangential migration: with probability tangential_migration_prob, a neuron moves to a randomly
          chosen neighboring fiber if its fiber's cell cycle state <= tangential_migration_cell_cycle,
          it is at height IMZ_START-IMZ_END, and its fiber is not trapped
      (3) all neurons climb by one bin
      (4) cell cycles advance; a fiber whose cycle wraps around generates a neuron with probability
          cell_generation_prob[t], and a second neuron (via an IP) with probability IP_ratio

    Returns the number of neurons on each fiber, the list of neurons, and, if record_cell_cycle is True,
    the cell cycle state of every fiber at every bin (N_TIME_BINS x n_fibers; Fig. 3f, g).
    """
    neuron_list = []
    cell_cycle_history = [] if record_cell_cycle else None

    for timebin in range(N_TIME_BINS):
        for neuron in neuron_list:
            if neuron.position >= trap_effective_height + IMZ_END:
                neuron.radial_fiber.trap = True

        for neuron in neuron_list:
            if (tangential_migration_prob > np.random.rand() and
                    neuron.radial_fiber.cellcycle <= tangential_migration_cell_cycle and
                    neuron.position <= IMZ_END and
                    neuron.position >= IMZ_START and
                    neuron.radial_fiber.trap == False):
                neuron.radial_fiber = np.random.choice(neuron.radial_fiber.adjacent_radialfibers)

        for neuron in neuron_list:
            neuron.position += 1

        for radial_fiber in radial_fiber_list:
            radial_fiber.pass_time(cell_cycle_hours, BINS_PER_HOUR)

            # The cycle wrapped around within this bin -> division
            if radial_fiber.cellcycle < 1 / (cell_cycle_hours * BINS_PER_HOUR):
                if cell_generation_prob[timebin] > np.random.rand():
                    neuron_list.append(Neuron(radial_fiber, 0))
                    if IP_ratio > np.random.rand():
                        neuron_list.append(Neuron(radial_fiber, 0))

        if record_cell_cycle:
            cell_cycle_history.append([rf.cellcycle for rf in radial_fiber_list])

    cell_count = np.zeros(len(radial_fiber_list))
    for neuron in neuron_list:
        cell_count[neuron.radial_fiber.id] += 1

    if record_cell_cycle:
        return cell_count, neuron_list, np.array(cell_cycle_history)
    return cell_count, neuron_list


def celltype_ratio(cell_count, n_type0=N_TYPE0):
    """Mean number of neurons per type-1 fiber / per type-0 fiber."""
    return np.average(cell_count[n_type0:]) / np.average(cell_count[:n_type0])


def isolated_fiber_ratio(cell_count):
    """Fraction of fibers with no neurons attached."""
    return np.mean(cell_count == 0)


# ----------------------------------------------------------------------------
# Rescaling to experimental coordinates and relative density (Extended Data Fig. 7)
# ----------------------------------------------------------------------------
RGC_TARGET_SPACING_UM = 12.0   # spacing between adjacent type-0 (brown) RGCs
RGC_MOVE_DISTANCE_UM = 3.0     # every fiber is displaced by exactly 3 um in a random direction
NEURON_MOVE_DISTANCE_UM = 3.0  # every neuron is displaced by exactly 3 um from its (displaced) fiber
DISTANCE_BIN_WIDTH_UM = 3.0
DISTANCE_BIN_MAX_UM = 30.0
DISTANCE_BIN_EDGES = np.arange(0.0, DISTANCE_BIN_MAX_UM + DISTANCE_BIN_WIDTH_UM, DISTANCE_BIN_WIDTH_UM)


def sample_fixed_displacements(rng, count, distance_um):
    if count == 0 or distance_um <= 0:
        return np.zeros((count, 2), dtype=float)
    theta = rng.uniform(0, 2 * np.pi, size=count)
    return np.column_stack((distance_um * np.cos(theta), distance_um * np.sin(theta)))


def scaled_coordinates(radial_fiber_list, neuron_list, rng):
    """Rescale fibers to 12 um spacing, perturb fibers and neurons by 3 um (Extended Data Fig. 7a, b).

    Returns (fiber_xy, neuron_xy, neuron_fiber_ids) in um.
    """
    fiber_xy = np.array([rf.position for rf in radial_fiber_list], dtype=float) * RGC_TARGET_SPACING_UM
    fiber_xy += sample_fixed_displacements(rng, len(fiber_xy), RGC_MOVE_DISTANCE_UM)
    neuron_fiber_ids = np.array([neuron.radial_fiber.id for neuron in neuron_list], dtype=int)
    neuron_xy = fiber_xy[neuron_fiber_ids].copy()
    neuron_xy += sample_fixed_displacements(rng, len(neuron_xy), NEURON_MOVE_DISTANCE_UM)
    return fiber_xy, neuron_xy, neuron_fiber_ids


def relative_density(fiber_xy, neuron_xy, bin_edges=DISTANCE_BIN_EDGES):
    """Relative density of neurons as a function of tangential distance (Extended Data Fig. 7c, d).

    Neurons farther than the largest bin edge from the border of the fiber field are used as references.
    Returns (mean neuron count per reference neuron in each ring,
             density normalized by the mean density within the largest bin edge).
    """
    zero_bins = np.zeros(len(bin_edges) - 1, dtype=float)
    if len(fiber_xy) == 0 or len(neuron_xy) == 0:
        return zero_bins, zero_bins
    max_um = bin_edges[-1]
    ring_areas = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)

    x_min, y_min = fiber_xy.min(axis=0)
    x_max, y_max = fiber_xy.max(axis=0)
    dist_to_border = np.minimum(
        np.minimum(neuron_xy[:, 0] - x_min, x_max - neuron_xy[:, 0]),
        np.minimum(neuron_xy[:, 1] - y_min, y_max - neuron_xy[:, 1]),
    )
    interior_indices = np.where(dist_to_border >= max_um)[0]
    if len(interior_indices) == 0:
        return zero_bins, zero_bins

    diff = neuron_xy[None, :, :] - neuron_xy[interior_indices][:, None, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    dist_mat[np.arange(len(interior_indices)), interior_indices] = np.nan  # exclude self
    dist_values = dist_mat[~np.isnan(dist_mat)].ravel()

    hist_total, _ = np.histogram(dist_values, bins=bin_edges)
    mean_count_per_neuron = hist_total / len(interior_indices)
    density = mean_count_per_neuron / np.maximum(ring_areas, 1e-12)
    mean_density_within_max = np.sum(mean_count_per_neuron) / max(np.pi * max_um ** 2, 1e-12)
    normalized_density = density / max(mean_density_within_max, 1e-12)
    return mean_count_per_neuron, normalized_density
