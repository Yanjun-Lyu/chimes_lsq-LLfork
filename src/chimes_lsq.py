import sys
import numpy
import scipy.linalg
import math as m
import subprocess
import os
import argparse
import contextlib
import random


from numpy        import *
from numpy.linalg import lstsq
from numpy.linalg import LinAlgError
from datetime     import *
from subprocess   import call
import scipy.sparse.linalg as spla
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import gmres
import multiprocessing as mp

try:
    from pyswarms.single import GlobalBestPSO
    PYSWARMS_AVAILABLE = True
except ImportError:
    PYSWARMS_AVAILABLE = False
    GlobalBestPSO = None
    print("! Warning: pyswarms not available. Install with: pip install pyswarms")

try:
    import deap
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    print("! Warning: DEAP not available. Install with: pip install deap")


#############################################
#############################################
# Particle Swarm Optimization (PSO) Solver
#############################################
#############################################

def pso_solve(A_file,
              b_file,
              weights_file=None,
              swarm_size=80,
              iters=200,
              inertia=0.7298,
              c1=1.49618,
              c2=1.49618,
              l2_reg=1.0e-4,
              nlines=None,
              n_params=None,
              num_cores=1,
              seed=None,
              init_scale=0.1,
              bounds=(-1.0, 1.0),
              vmax_frac=0.2,
              tol=1.0e-10,
              stall_iters=30):
    """
    Memory-efficient Particle Swarm Optimization for (optionally weighted) linear least-squares with L2 regularization.
    Uses pyswarms library for efficient distributed PSO computation.

    Objective for a parameter vector x is:
        fitness(x) = sum_i ( w_i * (A_i dot x - b_i) )^2 + l2_reg * ||x||^2

    Reads A, b, and weights directly from disk in chunks — no large arrays in memory.
    Parallel across num_cores using multiprocessing (same chunking strategy as GA).
    """
    if not PYSWARMS_AVAILABLE:
        raise ImportError("pyswarms library is required. Install with: pip install pyswarms")
    
    if seed is not None:
        numpy.random.seed(seed)

    if nlines is None or n_params is None:
        raise ValueError("nlines and n_params must be provided")

    lo, hi = float(bounds[0]), float(bounds[1])
    if hi <= lo:
        raise ValueError("PSO bounds must satisfy hi > lo")

    # Determine if pyswarms will handle parallelization
    # If pyswarms uses n_processes > 1, disable internal multiprocessing to avoid nested parallelization
    pyswarms_parallel = (num_cores > 1)
    use_internal_mp = not pyswarms_parallel  # Only use internal MP if pyswarms isn't parallelizing
    
    if pyswarms_parallel:
        print(f"! Pyswarms parallelization enabled ({num_cores} processes) - disabling internal multiprocessing")
    else:
        print(f"! Pyswarms running serially - internal multiprocessing {'enabled' if num_cores > 1 else 'disabled'}")

    # Create a wrapper function that captures the parameters for pyswarms
    # Use functools.partial or lambda to create a callable that pyswarms can pickle
    from functools import partial
    objective_function = partial(
        pso_objective_function,
        A_file=A_file,
        b_file=b_file,
        weights_file=weights_file,
        nlines=nlines,
        l2_reg=l2_reg,
        num_cores=num_cores,
        use_internal_mp=use_internal_mp
    )

    # Initialize optimizer with pyswarms
    # Set up options for PSO
    options = {'c1': c1, 'c2': c2, 'w': inertia}
    
    # Calculate velocity clamp if specified
    velocity_clamp_tuple = None
    if vmax_frac is not None:
        box = (hi - lo)
        vmax = vmax_frac * box
        velocity_clamp_tuple = (-vmax, vmax)
    
    # Create bounds array with explicit shape (2, n_params) to avoid IndexError
    # pyswarms periodic boundary handler requires bounds to have shape (2, dimensions)
    # where first row is lower bounds and second row is upper bounds
    bounds_array = numpy.array([[lo] * n_params, [hi] * n_params])
    
    # Set initial positions with init_scale if specified
    init_pos = None
    if init_scale is not None:
        init_pos = numpy.random.randn(swarm_size, n_params) * init_scale
        init_pos = numpy.clip(init_pos, lo, hi)
    
    # Create optimizer instance
    # Using GlobalBestPSO for standard PSO behavior
    # Use explicit bounds array to ensure correct shape for boundary handlers
    optimizer = GlobalBestPSO(
        n_particles=swarm_size,
        dimensions=n_params,
        options=options,
        bounds=bounds_array,  # Use explicit array format to avoid shape issues
        init_pos=init_pos,
        velocity_clamp=velocity_clamp_tuple
    )

    print(f"! PSO started (using pyswarms): swarm_size={swarm_size}, iters={iters}, cores={num_cores}, bounds=[{lo},{hi}]")
    
    # Run optimization
    # pyswarms handles parallelization internally via n_processes
    # When pyswarms parallelizes, internal multiprocessing is disabled to avoid nested parallelization
    cost, pos = optimizer.optimize(
        objective_function,
        iters=iters,
        n_processes=num_cores if pyswarms_parallel else None,
        verbose=False
    )
    
    # Print progress from cost history if available
    if hasattr(optimizer, 'cost_history') and optimizer.cost_history:
        # Print periodic updates from history
        history_len = len(optimizer.cost_history)
        for i in range(0, history_len, max(1, history_len // 5)):  # Print ~5 updates
            if i < history_len:
                print(f"! PSO iter {i:4d} | best fitness = {optimizer.cost_history[i]:.10e}")
        # Always print final
        if history_len > 0:
            print(f"! PSO iter {history_len-1:4d} | best fitness = {optimizer.cost_history[-1]:.10e}")
    
    print(f"! PSO finished. Final best fitness = {cost:.10e}")
    
    return pos


#############################################
#############################################
# Nonlinear residual and loss functions
#############################################
#############################################

def nonlinear_residual(x, A, b):
    """
    Nonlinear residual for force matching:
    r = A x - b + nonlinear correction
    """
    return A @ x - b + numpy.sin(x)


def nonlinear_loss(x, A, b, alpha=0.0):
    """
    Scalar loss for L-BFGS
    """
    r = nonlinear_residual(x, A, b)
    return 0.5 * numpy.dot(r, r) + 0.5 * alpha * numpy.dot(x, x)

#############################################
# Genetic Algorithm Solver
#############################################
#############################################
# Genetic Algorithm Solver — Memory Efficient & Parallel
#############################################

# ------------------------------------------------------------------
# Worker function: compute partial fitness for a chunk of rows
# Must be module-level for multiprocessing pickling
# ------------------------------------------------------------------
def chunk_fitness(start, end, A_file, b_file, weights_file, population):
    n_ind = population.shape[0]
    partial_scores = numpy.zeros(n_ind, dtype=float)

    # Open files — each process opens its own handles (safe)
    with open(A_file, 'r') as Af, \
         open(b_file, 'r') as bf, \
         (open(weights_file, 'r') if weights_file else contextlib.nullcontext()) as wf:

        # Skip to starting line
        for _ in range(start):
            Af.readline()
            bf.readline()
            if weights_file:
                wf.readline()

        for _ in range(start, end):
            A_line = Af.readline()
            b_line = bf.readline()
            if not A_line or not b_line:
                break  # safety

            Arow = numpy.fromstring(A_line, sep=' ', dtype=float)
            bi = float(b_line.strip())
            if weights_file:
                wi = float(wf.readline().strip())
            else:
                wi = 1.0

            # residuals = Arow @ individual - bi   for all individuals
            residuals = Arow @ population.T - bi
            residuals *= wi
            partial_scores += residuals ** 2

    return partial_scores


# ------------------------------------------------------------------
# Module-level objective function for pyswarms
# Must be module-level for multiprocessing pickling
# ------------------------------------------------------------------
def pso_objective_function(swarm_positions, A_file, b_file, weights_file, nlines, l2_reg, num_cores, use_internal_mp):
    """
    Objective function for pyswarms (module-level for pickling).
    swarm_positions: array of shape (n_particles, n_dimensions)
    Returns: array of shape (n_particles,) with fitness values
    """
    reg_term = l2_reg * numpy.sum(swarm_positions ** 2, axis=1)
    
    # Use internal multiprocessing only if pyswarms is not handling parallelization
    if not use_internal_mp or num_cores <= 1:
        # Serial evaluation (pyswarms will parallelize across particles if enabled)
        scores = chunk_fitness(0, nlines, A_file, b_file, weights_file, swarm_positions)
    else:
        # Internal multiprocessing for chunking (only when pyswarms is serial)
        chunk_size = nlines // num_cores
        chunks = []
        for i in range(num_cores):
            s = i * chunk_size
            e = (i + 1) * chunk_size if i < num_cores - 1 else nlines
            chunks.append((s, e))

        with mp.Pool(processes=num_cores) as pool:
            results = pool.starmap(
                chunk_fitness,
                [(s, e, A_file, b_file, weights_file, swarm_positions) for s, e in chunks]
            )
        scores = numpy.zeros(swarm_positions.shape[0], dtype=float)
        for r in results:
            scores += r
    
    return scores + reg_term


def ga_solve(A_file,
             b_file,
             weights_file=None,
             pop_size=100,
             generations=1000,
             mutation_rate=0.05,
             crossover_rate=0.8,
             elite_frac=0.1,
             l2_reg=1.0e-4,
             nlines=None,
             n_params=None,
             num_cores=1,
             seed=None):
    """
    Memory-efficient Genetic Algorithm for linear least-squares with optional weighting and L2 regularization.
    Uses DEAP library for efficient distributed GA computation.
    Reads A, b, and weights directly from disk in chunks — no large arrays in memory.
    """
    if not DEAP_AVAILABLE:
        raise ImportError("DEAP library is required. Install with: pip install deap")
    
    if seed is not None:
        numpy.random.seed(seed)
        random.seed(seed)

    if nlines is None or n_params is None:
        raise ValueError("nlines and n_params must be provided")

    # Determine if DEAP will handle parallelization
    # If DEAP uses multiple processes, disable internal multiprocessing to avoid nested parallelization
    deap_parallel = (num_cores > 1)
    use_internal_mp = not deap_parallel  # Only use internal MP if DEAP isn't parallelizing
    
    if deap_parallel:
        print(f"! DEAP parallelization enabled ({num_cores} processes) - disabling internal multiprocessing")
    else:
        print(f"! DEAP running serially - internal multiprocessing {'enabled' if num_cores > 1 else 'disabled'}")

    # Create fitness class and individual class (only if they don't exist)
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize fitness
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    # Create toolbox
    toolbox = base.Toolbox()
    
    # Register attribute generator (individual genes)
    toolbox.register("attr_float", numpy.random.randn)  # Will be scaled in individual creation
    
    # Register individual creator
    def create_individual():
        """Create an individual with small initial values"""
        ind = creator.Individual(numpy.random.randn(n_params) * 0.1)
        return ind
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define fitness evaluation function
    def evaluate_individual(individual):
        """
        Evaluate fitness of a single individual.
        individual: DEAP Individual (list-like)
        Returns: tuple (fitness,) for DEAP
        """
        # Convert individual to numpy array (2D for chunk_fitness compatibility)
        ind_array = numpy.array(individual).reshape(1, -1)
        
        # Compute fitness using chunk_fitness
        reg_term = l2_reg * numpy.sum(ind_array ** 2)
        
        if not use_internal_mp or num_cores <= 1:
            # Serial evaluation (DEAP will parallelize across individuals if enabled)
            # chunk_fitness returns array of shape (1,), so take first element
            score = float(chunk_fitness(0, nlines, A_file, b_file, weights_file, ind_array)[0])
        else:
            # Internal multiprocessing for chunking (only when DEAP is serial)
            chunk_size = nlines // num_cores
            chunks = []
            for i in range(num_cores):
                s = i * chunk_size
                e = (i + 1) * chunk_size if i < num_cores - 1 else nlines
                chunks.append((s, e))

            with mp.Pool(processes=num_cores) as pool:
                results = pool.starmap(
                    chunk_fitness,
                    [(s, e, A_file, b_file, weights_file, ind_array) for s, e in chunks]
                )
            # Sum results (each result is array of shape (1,))
            score = float(sum(r[0] for r in results))
        
        return (score + reg_term,)

    # Register fitness evaluation
    toolbox.register("evaluate", evaluate_individual)
    
    # Register genetic operators
    toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=mutation_rate)
    toolbox.register("select", tools.selTournament, tournsize=2)  # Tournament selection

    # Create initial population
    population = toolbox.population(n=pop_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Extract initial best fitness
    initial_best = min(fit[0] for fit in fitnesses)
    print(f"! GA started (using DEAP): pop_size={pop_size}, generations={generations}, cores={num_cores}")
    print(f"! Initial best fitness = {initial_best:.6e}")

    # Statistics for tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)

    # Hall of fame for elites
    hof = tools.HallOfFame(maxsize=max(1, int(elite_frac * pop_size)))

    # Setup parallelization if enabled
    if deap_parallel:
        # DEAP handles parallelization internally
        pool = mp.Pool(processes=num_cores)
        toolbox.register("map", pool.map)
    else:
        # Use standard map (serial)
        toolbox.register("map", map)

    # Run evolution
    try:
        population, logbook = algorithms.eaSimple(
            population,
            toolbox,
            cxpb=crossover_rate,      # Crossover probability
            mutpb=mutation_rate,      # Mutation probability
            ngen=generations,         # Number of generations
            stats=stats,
            halloffame=hof,
            verbose=False
        )
    finally:
        # Clean up pool if created
        if deap_parallel:
            pool.close()
            pool.join()

    # Get best individual from hall of fame
    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]
    
    # Print progress from logbook
    if logbook:
        gen_entries = logbook.select("gen", "min")
        for i in range(0, len(gen_entries), max(1, len(gen_entries) // 5)):  # Print ~5 updates
            if i < len(gen_entries):
                gen, min_fit = gen_entries[i]
                print(f"! GA generation {gen:4d} | best fitness = {min_fit:.10e}")
        # Always print final
        if len(gen_entries) > 0:
            gen, min_fit = gen_entries[-1]
            print(f"! GA generation {gen:4d} | best fitness = {min_fit:.10e}")

    print(f"! GA finished. Final best fitness = {best_fitness:.10e}")
    
    # Return as numpy array for compatibility
    return numpy.array(best_individual)



#############################################
#############################################
# Main
#############################################
#############################################

def main():
    
    loc = os.path.dirname(os.path.abspath(__file__))
    
    
    #############################################
    # Define arguments supported by the lsq code
    #############################################
        
    parser = argparse.ArgumentParser(description='Least-squares force matching based on output of chimes_lsq')

    parser.add_argument("--A",                    type=str,      default='A.txt',         help='A (derivative) matrix') 
    parser.add_argument("--algorithm",            type=str,      default='svd',           help='fitting algorithm: svd, fast_svd, ridge, lasso, lbfgs, gauss_newton, jfnk, ga, pso')
    parser.add_argument("--dlasso_dlars_path",    type=str     , default=loc+'/../contrib/dlars/src/',              help='Path to DLARS and/or DLASSO solver')
    parser.add_argument("--alpha",                type=float,    default=1.0e-04,         help='Lasso regularization')
    parser.add_argument("--b",                    type=str,      default='b.txt',         help='b (force) file')
    parser.add_argument("--cores",                type=int,      default=8,               help='DLARS number of cores')
    parser.add_argument("--eps",                  type=float,    default=1.0e-05,         help='svd regularization')
    parser.add_argument("--header",               type=str,      default='params.header', help='parameter file header')
    parser.add_argument("--map",                  type=str,      default='ff_groups.map', help='parameter file map')
    parser.add_argument("--nodes",                type=int,      default=1,               help='DLARS number of nodes')
    parser.add_argument("--mpistyle",             type=str,      default="srun",          help='Command used to run an MPI job, e.g. srun, ibrun, mpriun, etc')
    parser.add_argument("--normalize",            type=str2bool, default=False,           help='Normalize DLARS calculation')
    parser.add_argument("--read_output",          type=str2bool, default=False,           help='Read output from previous DLARS run')
    parser.add_argument("--restart_dlasso_dlars", type=str,      default="",              help='Determines whether dlasso or dlars job will be restarted. Argument is the restart file name ')
    parser.add_argument("--split_files",          type=str2bool, default=False,           help='LSQ code has split A matrix output.  Works DLARS.')
    parser.add_argument("--test_suite",           type=str2bool, default=False,           help='output for test suite')
    parser.add_argument("--weights",              type=str,      default="None",          help='weight file')
    parser.add_argument("--active",               type=str2bool, default=False,           help='is this a DLARS/DLASSO run from the active learning driver?')
    parser.add_argument("--folds",type=int, default=4,help="Number of CV folds")
    parser.add_argument("--hyper_sets",           type=str2bool, default=False,           help='Are you trying to fit a model with multiple hyperparameter sets?')
    parser.add_argument("--seed",                 type=int,      default=None,            help='Random seed for GA/PSO (and any stochastic solver)')

    # PSO options (used when --algorithm pso)
    parser.add_argument("--pso_swarm_size",       type=int,      default=80,              help='PSO swarm size')
    parser.add_argument("--pso_iters",            type=int,      default=200,             help='PSO iterations')
    parser.add_argument("--pso_inertia",          type=float,    default=0.7298,          help='PSO inertia weight (w)')
    parser.add_argument("--pso_c1",               type=float,    default=1.49618,         help='PSO cognitive coefficient (c1)')
    parser.add_argument("--pso_c2",               type=float,    default=1.49618,         help='PSO social coefficient (c2)')
    parser.add_argument("--pso_init_scale",       type=float,    default=0.1,             help='PSO init stddev (positions ~ N(0, scale))')
    parser.add_argument("--pso_bounds",           type=float,    nargs=2, default=[-1.0, 1.0], help='PSO box bounds: min max (applied to all params)')
    parser.add_argument("--pso_vmax_frac",        type=float,    default=0.2,             help='PSO max velocity as fraction of (max-min) bound range')
    parser.add_argument("--pso_tol",              type=float,    default=1.0e-10,         help='PSO improvement tolerance for early stopping')
    parser.add_argument("--pso_stall_iters",      type=int,      default=30,              help='PSO early stop after this many stall iterations')
    # Actually parse the arguments

    args        = parser.parse_args()
    print(args.hyper_sets)
    dlasso_dlars_path = args.dlasso_dlars_path

    #############################################
    # Import sklearn modules, if needed
    #############################################

    # Algorithms requiring sklearn.
    sk_algos = ["lasso", "ridge", "lassolars", "lars", "ridgecv"] ;

    if args.algorithm in sk_algos:
        from sklearn import linear_model
        from sklearn import preprocessing
        
    #############################################
    # Read weights, if used
    #############################################        

        
    WEIGHTS = None    
    if args.weights == "None":
        DO_WEIGHTING = False 
    else:
        DO_WEIGHTING = True
        if not args.split_files:
            WEIGHTS = numpy.genfromtxt(args.weights, dtype='float')

    #################################
    # Process A and b matrices — smart loading
    #################################

    # Always load b — it's 1D and small
    b = numpy.genfromtxt(args.b, dtype='float')
    nlines = b.shape[0]

    # Special cases: active learning or dlasso without split files
    if (args.active and not args.split_files) or (args.algorithm == "dlasso" and not args.split_files):
        A = numpy.zeros((1,1), dtype=float)  # Dummy
        np = "undefined"
    
    # Case: split_files or read_output → don't load A, get dims from file
    elif args.split_files or args.read_output:
        if not args.read_output:
            with open("dim.0000.txt", "r") as dimf:
                line = next(dimf)
                np, nstart, nend, nlines_from_dim = (int(x) for x in line.split())
            A = numpy.zeros((1,1), dtype=float)  # Dummy
        else:
            np = "undefined"
            A = None  # Not needed
        # Note: nlines already from b
    
    # Default case: not split, not read_output, not special → load full A
    # BUT: if algorithm is 'ga', we SKIP loading full A to save memory
    elif args.algorithm in ["ga", "pso"]:
        print("! GA/PSO algorithm detected: skipping full load of A matrix (memory-efficient mode)")
        A = None  # Will read rows on-demand in ga_solve
        
        # Infer number of parameters (np) from first row of A.txt
        with open(args.A) as f:
            first_line = f.readline().strip()
            if not first_line:
                print("Error: A file is empty")
                exit(1)
            np = len(first_line.split())
        print(f"! Inferred number of parameters (np) = {np} from first row of {args.A}")

    else:
        # Original behavior: load full A for all other algorithms
        print("! Loading full A matrix into memory...")
        A = numpy.genfromtxt(args.A, dtype='float')
        
        if A.shape[0] != nlines:
            print("Error: A and b have mismatched number of rows")
            exit(1)
        
        np = A.shape[1]

        if np > nlines:
            print("Error: number of variables > number of equations")
            exit(1)

    # Sanity check weights length
    if DO_WEIGHTING and not args.split_files and WEIGHTS is not None:
        if WEIGHTS.shape[0] != nlines:
            print("Error: Wrong number of lines in WEIGHTS file")
            exit(1)

    #################################
    # Apply weighting to A and b (only if A was loaded)
    #################################

    weightedA = None
    weightedb = None

    if DO_WEIGHTING and not args.split_files and not args.active and args.algorithm != "dlasso":
        if A is None:
            print("! Warning: Weighting requested but A not loaded (GA/PSO mode) — skipping weightedA creation")
        else:
            weightedA = numpy.zeros_like(A)
            weightedb = numpy.zeros_like(b)
            for i in range(nlines):
                weightedA[i] = A[i] * WEIGHTS[i]
                weightedb[i] = b[i] * WEIGHTS[i]

    ################################                
    # Header for output
    ################################
    
    print("! Date ", date.today())
    print("!")

    if np != "undefined":
        print("! Number of variables            = ", np)

    print("! Number of equations            = ", nlines)

    
                
    #################################
    # Solve the matrix equation
    #################################

    if args.algorithm == 'svd':
        
        # Make the scipy call
        
        print ('! svd algorithm used')
        try:
            print(args.hyper_sets)
            if DO_WEIGHTING and args.hyper_sets is False: # Then it's OK to overwrite weightedA.  It is not used to calculate y (predicted forces) below.
                U,D,VT = scipy.linalg.svd(weightedA,overwrite_a=True)
                Dmat   = array((transpose(weightedA)))
                dmax = 0.0
                
            elif DO_WEIGHTING and args.hyper_sets is True :            #  Then do not overwrite A.  It is used to calculate y (predicted forces) below.
                
                min_shape = min(weightedA.shape)
                k = min(max(1, min_shape // 10), min_shape)
                U, D, VT = spla.svds(weightedA, k=k)
                Dmat = numpy.zeros((len(D), len(D)))
                dmax = numpy.max(numpy.abs(D))
                
            elif args.weights == "None" and args.hyper_sets is True :            #  Then do not overwrite A.  It is used to calculate y (predicted forces) below.
                min_shape = min(A.shape)
                k = min(max(1, min_shape // 10), min_shape)
                U, D, VT = spla.svds(A, k=k)
                Dmat = numpy.zeros((len(D), len(D)))
                dmax = numpy.max(numpy.abs(D))
                
            else :   # Previous Method
                min_shape = min(A.shape)
                k = min(max(1, min_shape // 10), min_shape)
                U, D, VT = spla.svds(A, k=k)
                Dmat = numpy.zeros((len(D), len(D)))
                dmax = numpy.max(numpy.abs(D))
                
        except LinAlgError:
            sys.stderr.write("SVD algorithm failed")
            exit(1)
            
        # Process output

        #dmax = numpy.max(numpy.abs(D))

        for i in range(0,len(Dmat)):
            if ( abs(D[i]) > dmax ) :
                dmax = abs(D[i])

            for j in range(0,len(Dmat[i])):
                Dmat[i][j]=0.0

        # Cut off singular values based on fraction of maximum value as per numerical recipes.
        
        eps=args.eps * dmax
        nvars = 0

        for i in range(0,len(D)):
            if abs(D[i]) > eps:
                Dmat[i][i]=1.0/D[i]
                nvars += 1

        print ("! eps (= args.eps*dmax)          =  %11.4e" % eps)        
        print ("! SVD regularization factor      = %11.4e" % args.eps)

        x=dot(transpose(VT),Dmat)

        if DO_WEIGHTING:
            x = dot(x,dot(transpose(U),weightedb))
        else:
            x = dot(x,dot(transpose(U),b))

    elif args.algorithm == 'ridge':
        print ('! ridge regression used')
        reg = linear_model.Ridge(alpha=args.alpha,fit_intercept=False)

        # Fit the data.
        reg.fit(A,b)

        x = reg.coef_
        nvars = np
        print ("! Ridge alpha = %11.4e" % args.alpha)
        
    elif args.algorithm == 'fast_ridge':
        print ('! fast ridge regression used')

        ATA = A.T @ A
        ATb = A.T @ b

        ATA_reg = ATA.copy()
        numpy.fill_diagonal(ATA_reg, ATA_reg.diagonal() + args.alpha)

        x = numpy.linalg.solve(ATA_reg, ATb)
        
        nvars = np
        print ("! Ridge alpha = %11.4e" % args.alpha)

    elif args.algorithm == 'ridgecv':
        alpha_ar = [1.0e-06, 3.2e-06, 1.0e-05, 3.2e-05, 1.0e-04, 3.2e-04, 1.0e-03, 3.2e-03]
        reg = linear_model.RidgeCV(alphas=alpha_ar,fit_intercept=False,cv=args.folds)
        reg.fit(A,b)
        print ('! ridge CV regression used')
        print ("! ridge CV alpha = %11.4e"  % reg.alpha_)
        x = reg.coef_
        nvars = np

    elif args.algorithm == 'lasso':
        
        # Make the sklearn call
        
        print ('! Lasso regression used')
        print ('! Lasso alpha = %11.4e' % args.alpha)
        reg   = linear_model.Lasso(alpha=args.alpha,fit_intercept=False,max_iter=100000)
        reg.fit(A,b)
        x     = reg.coef_
        np    = count_nonzero_vars(x)
        nvars = np

    elif args.algorithm == 'lassolars':
        
        # Make the sklearn call
        
        print ('! LARS implementation of LASSO used')
        print ('! LASSO alpha = %11.4e' % args.alpha)

        if DO_WEIGHTING:
            reg = linear_model.LassoLars(alpha=args.alpha,fit_intercept=False,fit_path=False,verbose=True,max_iter=100000, copy_X=False)
            reg.fit(weightedA,weightedb)
        else:
            reg = linear_model.LassoLars(alpha=args.alpha,fit_intercept=False,fit_path=False,verbose=True,max_iter=100000)
            reg.fit(A,b)
        x       = reg.coef_[0]
        np      = count_nonzero_vars(x)
        nvars   = np

    elif args.algorithm == 'dlars' or args.algorithm == 'dlasso' :
        
        # Make the DLARS or DLASSO call

        x,y = fit_dlars(dlasso_dlars_path, args.nodes, args.cores, args.alpha, args.split_files, args.algorithm, args.read_output, args.weights, args.normalize, args.A , args.b ,args.restart_dlasso_dlars, args.mpistyle)
        np = count_nonzero_vars(x)
        nvars = np

    elif args.algorithm == 'lbfgs':

        print('! L-BFGS nonlinear optimization used')

        if DO_WEIGHTING:
            A_use = weightedA
            b_use = weightedb
        else:
            A_use = A
            b_use = b

        x0 = zeros(A_use.shape[1])


        res = minimize(
            nonlinear_loss,
            x0,
            args=(A_use, b_use, args.alpha),
            method='L-BFGS-B',
            options={'maxiter': 500, 'disp': True}
        )
        x = res.x
        nvars = count_nonzero_vars(x)
        nvars = np


    elif args.algorithm == 'gauss_newton':

        print('! Gauss–Newton nonlinear least squares used')

        if DO_WEIGHTING:
            A_use = weightedA
            b_use = weightedb
        else:
            A_use = A
            b_use = b

        x0 = zeros(A_use.shape[1])

        res = least_squares(
            nonlinear_residual,
            x0,
            args=(A_use, b_use),
            method='trf',
            max_nfev=500
        )
        x = res.x
        nvars = count_nonzero_vars(x)
        nvars = np


    elif args.algorithm == 'jfnk':

        print('! Jacobian-Free Newton–Krylov used')

        if DO_WEIGHTING:
            A_use = weightedA
            b_use = weightedb
        else:
            A_use = A
            b_use = b


        def F(x):
            return nonlinear_residual(x, A_use, b_use)

        x0 = zeros(A_use.shape[1])

        try:
            x = newton_krylov(
                F,
                x0,
                method='gmres',
                f_tol=1e-8,
                maxiter=50
            )
        
        except Exception as e:
            print("JFNK failed:", e)
            exit(1)

        nvars = count_nonzero_vars(x)
        nvars = np
    
    #################################
    # Solve the matrix equation
    #################################
    elif args.algorithm == 'ga':
        print('! Genetic Algorithm solver used')
        if DO_WEIGHTING:
            w = WEIGHTS
        else:
            w = None
        if args.split_files:
            print("! Warning: split_files not fully supported for GA; assuming single A.txt")
        if A is not None and hasattr(A, 'shape') and A.shape[1] > 0:
            np = A.shape[1]
        else:
            with open(args.A) as f:
                first_line = f.readline().strip()
                if not first_line:
                    print("Error: Cannot read first line of A file or file is empty")
                    exit(1)
                np = len(first_line.split())
            print(f"! Inferred number of parameters (np) = {np} from first row of A file")
        nlines = len(b)  
        x = ga_solve(
            A_file=args.A,
            b_file=args.b,
            weights_file=args.weights if DO_WEIGHTING else None,
            pop_size=50,
            generations=100,
            mutation_rate=0.05,
            crossover_rate=0.8,
            elite_frac=0.1,
            l2_reg=args.alpha,
            nlines=nlines,
            n_params=np,
            num_cores=args.cores  
        )
        nvars = count_nonzero_vars(x)
        nvars = np 

    elif args.algorithm == 'pso':
        print('! Particle Swarm Optimization solver used')
        if args.split_files:
            print("! Warning: split_files not fully supported for PSO; assuming single A.txt")

        # Ensure np is inferred even if A was not loaded
        if A is not None and hasattr(A, 'shape') and A.shape[1] > 0:
            np = A.shape[1]
        else:
            with open(args.A) as f:
                first_line = f.readline().strip()
                if not first_line:
                    print("Error: Cannot read first line of A file or file is empty")
                    exit(1)
                np = len(first_line.split())
            print(f"! Inferred number of parameters (np) = {np} from first row of A file")

        nlines = len(b)
        bounds = (float(args.pso_bounds[0]), float(args.pso_bounds[1]))

        x = pso_solve(
            A_file=args.A,
            b_file=args.b,
            weights_file=args.weights if DO_WEIGHTING else None,
            swarm_size=args.pso_swarm_size,
            iters=args.pso_iters,
            inertia=args.pso_inertia,
            c1=args.pso_c1,
            c2=args.pso_c2,
            l2_reg=args.alpha,
            nlines=nlines,
            n_params=np,
            num_cores=args.cores,
            seed=args.seed,
            init_scale=args.pso_init_scale,
            bounds=bounds,
            vmax_frac=args.pso_vmax_frac,
            tol=args.pso_tol,
            stall_iters=args.pso_stall_iters
        )
        nvars = count_nonzero_vars(x)
        nvars = np

    else:
        print ("Unrecognized fitting algorithm") 
        exit(1)

    #################################
    # Process output from solver(s)
    #################################

    # If split_files, A is not read in ...This conditional should really be set by the algorithm, since many set  y themselves...  
    
    # y = 0.0
    print("DEBUG:", args.algorithm, args.split_files, args.read_output, args.active)
    if ( (not args.split_files) and (not args.read_output) and (not args.active ) and (args.algorithm != "dlasso") ):
        if A is not None:
            y=dot(A,x)
        elif args.algorithm in ["ga", "pso"]:
            # For GA mode, compute y on-demand by reading A file row-by-row
            print("! Computing y = A*x on-demand (GA/PSO mode, A not in memory)")
            y = numpy.zeros(len(b), dtype=float)
            with open(args.A, 'r') as Af:
                for i in range(len(b)):
                    A_line = Af.readline()
                    if not A_line:
                        break
                    Arow = numpy.fromstring(A_line, sep=' ', dtype=float)
                    y[i] = numpy.dot(Arow, x)
        else:
            print("! Warning: A is None but algorithm is not GA - cannot compute y")
            y = numpy.zeros(len(b), dtype=float)

    Z=0.0

    # Put calculated forces in force.txt
    
    yfile = open("force.txt", "w")
    
    for a in range(0,len(b)):
        Z = Z + (y[a] - b[a]) ** 2.0
        yfile.write("%13.6e\n"% y[a]) 

    bic = float(nlines) * log(Z/float(nlines)) + float(nvars) * log(float(nlines))

    #############################################
    # Setup output
    #############################################
    
    print ("! RMSE                           = %11.4e" % sqrt(Z/float(nlines)))
    print ("! max abs variable               = %11.4e" %  max(abs(x)))
    print ("! number of fitting vars         = ", nvars)
    print ("! Bayesian Information Criterion = %11.4e" % bic)
    if args.weights !="None":
        print ('! Using weighting file:            ',args.weights)
    print ("!")

    ####################################
    # Actually process the header file...
    ####################################

    hf = open(args.header ,"r").readlines()
    
    BREAK_COND = False
    
    EXCL_2B = []

    # Figure out whether we have triplets and/or quadruplets
    # Find the ATOM_TRIPS_LINE and ATOM_QUADS_LINE
    # Find the TOTAL_TRIPS and TOTAL_QUADS

    ATOM_TRIPS_LINE = 0
    ATOM_QUADS_LINE = 0
    TOTAL_TRIPS = 0
    TOTAL_QUADS = 0

    for i in range(0, len(hf)):
        print (hf[i].rstrip('\n'))
        TEMP = hf[i].split()
        
        if "EXCL_2B" in hf[i]:
            line = line.split()
            EXCL_2B = line[1:]

        if len(TEMP)>3:
            if (TEMP[2] == "TRIPLETS:"):
                TOTAL_TRIPS = TEMP[3]
                ATOM_TRIPS_LINE = i

                for j in range(i, len(hf)):
                    TEMP = hf[j].split()
                    if len(TEMP)>3:
                        if (TEMP[2] == "QUADRUPLETS:"):
                            print (hf[j].rstrip('\n'))
                            TOTAL_QUADS = TEMP[3]
                            ATOM_QUADS_LINE = j
                            BREAK_COND = True
                            break
            if (BREAK_COND):
                 break

    # 1. Figure out what potential type we have

    POTENTIAL = hf[5].split()
    POTENTIAL = POTENTIAL[1]

    print ("")

    print ("PAIR " + POTENTIAL + " PARAMS \n")

    # 2. Figure out how many coeffs each atom type will have

    SNUM_2B = 0
    SNUM_4B = 0

    if POTENTIAL == "CHEBYSHEV":
        
        TMP = hf[5].split()

        if len(TMP) >= 4:
            if len(TMP) >= 5:
                SNUM_4B = int(TMP[4])

            SNUM_2B = int(TMP[2])  
 

    # 3. Print out the parameters

    FIT_COUL = hf[1].split()
    FIT_COUL = FIT_COUL[1]

    ATOM_TYPES_LINE  = 7
    TOTAL_ATOM_TYPES = hf[ATOM_TYPES_LINE].split()
    TOTAL_ATOM_TYPES = int(TOTAL_ATOM_TYPES[2])
    ATOM_PAIRS_LINE  = ATOM_TYPES_LINE+2+TOTAL_ATOM_TYPES+2
    TOTAL_PAIRS      = hf[ATOM_PAIRS_LINE].split()
    TOTAL_PAIRS      = int(TOTAL_PAIRS[2])
    
    # Remove excluded 2b interactions from accounting
    
    TOTAL_PAIRS -= len(EXCL_2B) 

    A1 = ""
    A2 = ""

    P1 = ""
    P2 = ""
    P3 = ""

    # PAIRS, AND CHARGES

    # Figure out how many 3B parameters there are

    SNUM_3B   = 0
    ADD_LINES = 0
    COUNTED_COUL_PARAMS = 0 

    if (int(TOTAL_TRIPS) > 0):
        for t in range(0, int(TOTAL_TRIPS)):

            P1 = hf[ATOM_TRIPS_LINE+3+ADD_LINES].split()

            if P1[4] != "EXCLUDED:":
                SNUM_3B +=  int(P1[4])

                TOTL = P1[6]
                ADD_LINES += 5

                for i in range(0, int(TOTL)):
                    ADD_LINES += 1

    # Figure out how many 4B parameters there are

    SNUM_4B   = 0
    ADD_LINES = 0

    if (int(TOTAL_QUADS) > 0):
        for t in range(0, int(TOTAL_QUADS)):

            P1 = hf[ATOM_QUADS_LINE+3+ADD_LINES].split()

            #print "QUAD HEADER", P1
            if P1[7] != "EXCLUDED:":

                SNUM_4B +=  int(P1[7])

                TOTL = P1[9]

                ADD_LINES += 5

                for i in range(0,int(TOTL)):
                    ADD_LINES += 1

    for i in range(0,TOTAL_PAIRS):

        A1 = hf[ATOM_PAIRS_LINE+2+i+1].split()
        A2 = A1[2]
        A1 = A1[1]

        #print ("PAIRTYPE PARAMS: " + `i` + " " + A1 + " " + A2 + "\n")
        print ("PAIRTYPE PARAMS: " + str(i) + " " + A1 + " " + A2 + "\n")

        for j in range(0, int(SNUM_2B)):
            print ("%3d %21.13e" % (j,x[i*SNUM_2B+j]))

        if FIT_COUL == "true":
            print ("q_%s x q_%s %21.13e" % (A1,A2,x[TOTAL_PAIRS*SNUM_2B + SNUM_3B + SNUM_4B + i]))
            COUNTED_COUL_PARAMS += 1

        print (" ")

    # TRIPLETS

    ADD_LINES = 0
    ADD_PARAM = 0

    COUNTED_TRIP_PARAMS = 0

    if (int(TOTAL_TRIPS) > 0):
        print ("TRIPLET " + POTENTIAL + " PARAMS \n")

        TRIP_PAR_IDX = 0

        for t in range(0, int(TOTAL_TRIPS)):

            PREV_TRIPIDX = 0

            print ("TRIPLETTYPE PARAMS:")
            print ("  " + hf[ATOM_TRIPS_LINE+2+ADD_LINES].rstrip())

            P1 = hf[ATOM_TRIPS_LINE+3+ADD_LINES].split()

            #print "HEADER: ", P1

            V0 = P1[1] 
            V1 = P1[2]
            V2 = P1[3]

            if P1[4] == "EXCLUDED:" :
                print ("   PAIRS: " + V0 + " " + V1 + " " + V2 + " EXCLUDED:")
                ADD_LINES += 1
            else:
                UNIQ = P1[4]
                TOTL = P1[6].rstrip() 

                print ("   PAIRS: " + V0 + " " + V1 + " " + V2 + " UNIQUE: " + UNIQ + " TOTAL: " + TOTL)
                print ("     index  |  powers  |  equiv index  |  param index  |       parameter       ")
                print ("   ----------------------------------------------------------------------------")

                ADD_LINES += 3

                if(t>0):
                    ADD_PARAM += 1

                for i in range(0,int(TOTL)):
                    ADD_LINES += 1
                    LINE       = hf[ATOM_TRIPS_LINE+2+ADD_LINES].rstrip('\n')
                    LINE_SPLIT = LINE.split()

                    print ("%s %21.13e" % (LINE, x[TOTAL_PAIRS*SNUM_2B + TRIP_PAR_IDX+int(LINE_SPLIT[5])]))

                TRIP_PAR_IDX += int(UNIQ)
                COUNTED_TRIP_PARAMS += int(UNIQ)
                #print "COUNTED_TRIP_PARAMS", COUNTED_TRIP_PARAMS

            print ("")

            ADD_LINES += 2

    ADD_LINES = 0
    
    # QUADS    

    COUNTED_QUAD_PARAMS = 0
    if (int(TOTAL_QUADS) > 0):
        print ("QUADRUPLET " + POTENTIAL + " PARAMS \n")

        QUAD_PAR_IDX = 0

        for t in range(int(TOTAL_QUADS)):

            PREV_QUADIDX = 0

            #print "ATOM_QUADS_LINE " + str(ATOM_QUADS_LINE+2+ADD_LINES)

            P1 = hf[ATOM_QUADS_LINE+2+ADD_LINES].split()

            #print "P1 " + P1[1] + P1[2] + P1[3] + P1[4] + P1[5] + P1[6]

            print ("QUADRUPLETYPE PARAMS: " )
            print ("  " + hf[ATOM_QUADS_LINE+2+ADD_LINES].rstrip() )

            P1 = hf[ATOM_QUADS_LINE+3+ADD_LINES].split()

            #print P1 

            V0 = P1[1] 
            V1 = P1[2]
            V2 = P1[3]
            V3 = P1[4] 
            V4 = P1[5]
            V5 = P1[6]

            #print "UNIQUE: ", str(UNIQ)
            if P1[7] == "EXCLUDED:" :
                print ("   PAIRS: " + V0 + " " + V1 + " " + V2 + " " + V3 + " " + V4 + " " + V5 + " EXCLUDED: ")
                ADD_LINES += 1

            else:
                UNIQ = P1[7]
                TOTL = P1[9].rstrip() 

                print ("   PAIRS: " + V0 + " " + V1 + " " + V2 + " " + V3 + " " + V4 + " " + V5 + " UNIQUE: " + UNIQ + " TOTAL: " + TOTL)
                print ("     index  |  powers  |  equiv index  |  param index  |       parameter       ")
                print ("   ----------------------------------------------------------------------------")

                ADD_LINES += 3

                if(t>0):
                    ADD_PARAM += 1

                for i in range(0,int(TOTL)):
                    ADD_LINES += 1
                    LINE       = hf[ATOM_QUADS_LINE+2+ADD_LINES].rstrip('\n')
                    LINE_SPLIT = LINE.split()

                    UNIQ_QUAD_IDX = int(LINE_SPLIT[8])
                    #print 'UNIQ_QUAD_IDX', str(UNIQ_QUAD_IDX)

                    print ("%s %21.13e" % (LINE,x[TOTAL_PAIRS*SNUM_2B + COUNTED_TRIP_PARAMS + QUAD_PAR_IDX + UNIQ_QUAD_IDX]))

                QUAD_PAR_IDX += int(UNIQ)
                COUNTED_QUAD_PARAMS += int(UNIQ)

            print ("")

            ADD_LINES += 2

    # Remaining tidbids

    mapsfile=open(args.map,"r").readlines()

    print ("")

    for i in range(0,len(mapsfile)):
        print (mapsfile[i].rstrip('\n'))

    print ("")

    total_params = TOTAL_PAIRS * SNUM_2B + COUNTED_TRIP_PARAMS + COUNTED_QUAD_PARAMS + COUNTED_COUL_PARAMS 

    N_ENER_OFFSETS = int(hf[7].split()[2])

## Parameter count could be off by natom_types, if energies are included in the fit
    if (total_params != len(x)) and (len(x) != (total_params+N_ENER_OFFSETS)) :
        sys.stderr.write( "Error in counting parameters\n") 
        sys.stderr.write("len(x) " + str(len(x)) + "\n") 
        sys.stderr.write("TOTAL_PAIRS " + str(TOTAL_PAIRS) + "\n") 
        sys.stderr.write("SNUM_2B " + str(SNUM_2B) + "\n") 
        sys.stderr.write("COUNTED_TRIP_PARAMS " + str(COUNTED_TRIP_PARAMS) + "\n") 
        sys.stderr.write("COUNTED_QUAD_PARAMS " + str(COUNTED_QUAD_PARAMS) + "\n")
        sys.stderr.write("COUNTED_COUL_PARAMS " + str(COUNTED_COUL_PARAMS) + "\n")
        exit(1)


    if len(x) == (total_params+N_ENER_OFFSETS):
        print ("NO ENERGY OFFSETS: ", N_ENER_OFFSETS)
    
        for i in range(N_ENER_OFFSETS):
            print ("ENERGY OFFSET %d %21.13e" % (i+1,x[total_params+i]))

    if args.test_suite:
        test_suite_params=open("test_suite_params.txt","w")		
        for i in range(0,len(x)):
            phrase = "%5d %21.13e\n" % (i,x[i])
            test_suite_params.write(phrase)
        test_suite_params.close()

    print ("ENDFILE")
    return 0

#############################################
#############################################
# Small helper functions
#############################################
#############################################

def is_number(s):
# Test if s is a number.
    try:
        float(s)
        return True
    except ValueError:
        return False


def str2bool(v):
## Convert string to bool.  Used in command argument parsing.
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")
                       

def count_nonzero_vars(x):
    np = 0
    for i in range(0, len(x)):
        if ( abs(x[i]) > 1.0e-05 ):
            np = np + 1
    return np


#############################################
#############################################
# DLARS wrapper
#############################################
#############################################

def fit_dlars(dlasso_dlars_path, nodes, cores, alpha, split_files, algorithm, read_output, weights, normalize, A , b, restart_dlasso_dlars, mpistyle):

    # Use the Distributed LARS/LASSO fitting algorithm.  Returns both the solution x and
    # the estimated force vector A * x, which is read from Ax.txt.    
    
    if dlasso_dlars_path == '':
        print ("ERROR: DLARS/DLASSO  path not provided.")
        print ("Please run again with --dlasso_dlars_path </absolute/path/to/dlars/dlasso/src/>")
        exit(0)

    if algorithm == 'dlasso' :
        print ('! DLARS code for LASSO used')
    elif algorithm == 'dlars' :
        print ('! DLARS code for LARS used')
    else:
        print ("Bad algorithm in fit_dlars:" + algorithm)
        exit(1)
    print ('! DLARS alpha = %10.4e' % alpha)

    if not read_output:
    
        dlars_file = dlasso_dlars_path + "dlars"
        
        if os.path.exists(dlars_file):
	
            exepath = ""

            if mpistyle == "srun": 
                exepath = "srun -N " + str(nodes) + " -n " + str(cores) + " " + dlars_file
            elif mpistyle == "ibrun":
                exepath = "ibrun" + " " + dlars_file  
            else:
                print("Unrecognized mpistyle:", mpistyle, ". Recognized options are srun or ibrun")
           
            command = None

            command = exepath + " " + A  + " " + b + " dim.txt --lambda=" + str(alpha)

            #else:
            #    command = ("{0} A.txt b.txt dim.txt --lambda={1}".format(exepath, alpha))

            if ( split_files ) :
                command = command + " --split_files"
            if ( algorithm == 'dlars' ):
                command = command + " --algorithm=lars"
            elif ( algorithm == 'dlasso' ):
                command = command + " --algorithm=lasso"

            if ( weights != 'None' ):
                command = command + " --weights=" + weights

            if ( normalize ):
                command = command + " --normalize=y" 
            else:
                command = command + " --normalize=n" 

            if restart_dlasso_dlars != "":
                print ("Will run a dlars/dlasso restart job with file:", restart_dlasso_dlars)

                command = command + " --restart=" + restart_dlasso_dlars
                
            command = command +  " >& dlars.log"

            print("! DLARS run: " + command + "\n")

            if ( os.system(command) != 0 ) :
                print(command + " failed")
                sys.exit(1)
        else:
            print (dlars_file + " does not exist")
            sys.exit(1)
    else:
        print ("! Reading output from prior DLARS calculation")

    x = numpy.genfromtxt("x.txt", dtype='float')
    y = numpy.genfromtxt("Ax.txt", dtype='float') 
    return x,y



# Python magic to allow having a main function definition.    
if __name__ == "__main__":
    mp.set_start_method('forkserver', force=True)
    main()
    














