import distopf as opf
from distopf.api import create_case
from distopf.matrix_models.matrix_bess.objectives import cp_obj_loss
from distopf.matrix_models.matrix_bess.solvers import cvxpy_solve

# from distopf.matrix_models.matrix_bess.lindist_mp import LinDistMP
from distopf.matrix_models.matrix_bess.base_mp import LinDistBaseMP
from time import perf_counter

import cProfile
import pstats
from pstats import SortKey

profiler = cProfile.Profile()
profiler.enable()

t0 = perf_counter()
case = create_case(
    data_path=opf.CASES_DIR / "csv" / "ieee123_30der", n_steps=1, start_step=12
)
# case.bus_data.v_max = 2
# case.bus_data.v_min = 0
case.schedules.default = 1
case.schedules.PV = 1
case.gen_data.control_variable = "PQ"
matrix_model = LinDistBaseMP(case=case)
matrix_model.build()
t1 = perf_counter()
results_matrix = cvxpy_solve(matrix_model, obj_func=cp_obj_loss)
t2 = perf_counter()
profiler.disable()
profiler.dump_stats("profile_matrix_results.prof")
print(f"Objective value: {results_matrix.fun}")
print(f"build time: {t1 - t0}")
print(f"solve time: {t2 - t1}")
stats = pstats.Stats("profile_matrix_results.prof")
stats.sort_stats(SortKey.TIME)
stats.print_stats(20)  # Top 20 functions by time
