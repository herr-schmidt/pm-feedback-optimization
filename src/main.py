from data_loader import DataLoader
from planners import HeuristicLBBDPlanner
from utils import SolutionVisualizer
from datetime import datetime, timedelta
from pprint import pprint
import bisect
import itertools

date_t_map = {}

# extracts a python dict of the form {date: list((gamma, patient_id, precedence))}
# needed for the PM2 phase
def extract_selected_patients_dict(solution, solver_input):
    starting_day = datetime(year=2022, month=9, day=5)

    selected_patients = {}

    for (i, k, t) in solution.x.keys():
        offset_weeks = (t - 1) // 5
        offset_days =  (t - 1) % 5
        day = starting_day + timedelta(days=offset_days, weeks=offset_weeks) # t - 1 since solver has t = 1 for the first day, and we want to include starting day
        patient_id = int(solver_input[None]["patientId"][i])

        gamma_id_precedence = (solution.gamma[i], patient_id, solution.precedence[i])
        print(gamma_id_precedence)

        if day in selected_patients:
            bisect.insort(selected_patients[day], gamma_id_precedence)
        else:
            selected_patients[day] = [gamma_id_precedence]

        # in order to easily retrieve t's when reaccessing solver's variables
        date_t_map[day] = t

    return selected_patients

# PM2
# return evaluation
def evaluate_slot(date, patients_ids):
    return 5

# compute, for each non-fixed slot, the permutations on which PM2 achieves the minimum overtime
# here selected_patients_dict should contain only the slots which were not fixed
def compute_best_permutations(selected_patients_dict):
    # will contain the tuple (permutation, permutation_evaluation) for each date
    best_permutations = {}
    
    for date in selected_patients_dict.keys():
        # extract a list of lists, according to the prerequisites of compute_ids_permutations (namely, a list of lists each one of which is
        # relative to a different precedence)
        current_precedence = 1
        current_precedence_partition = []
        id_partition_by_precedence = []
        for (_, id, precedence) in selected_patients_dict[date]:
            if precedence == current_precedence:
                current_precedence_partition.append(id)
            else:
                id_partition_by_precedence.append(current_precedence_partition)
                current_precedence = precedence
                current_precedence_partition = [id]

        # append last chunk
        id_partition_by_precedence.append(current_precedence_partition)

        permutations = compute_ids_permutations(id_partition_by_precedence)
        evaluations = [(permutation, evaluate_slot(date, permutation)) for permutation in permutations]
        best_permutation = min(evaluations, key=lambda e:e[1])

        best_permutations[date] = best_permutation

    return best_permutations


def extract_patient_ids(solution_dictionary):
    selected_patients = {}

    for date in solution_dictionary.keys():
        selected_patients[date] = list(map(lambda gamma_id_precedence: gamma_id_precedence[1], solution_dictionary[date]))

    return selected_patients

dl = DataLoader()
dl.load_input_file("../input/input.xlsx", predicted_operating_times=True)

solver_input = dl.generate_solver_input()
print(solver_input)

planner = HeuristicLBBDPlanner(timeLimit=600, gap = 0.0, iterations_cap=30, solver="cplex")
planner.solve_model(solver_input)

sv = SolutionVisualizer()
# sv.print_solution(solution)
print("Planned patients:" + str(sv.count_operated_patients(planner.solution)))
sv.plot_graph(planner.solution)

# compute utilization by pretending the computed schedule was succesfully implemented for
# all planned patients, but with the real-world time. WARNING: this cannot be done if more
# than 1 operating room is used and anesthesiae are taken into account.
def compute_real_world_utilization_and_overtime(solution, actual_times):
    utilizations = {(k, t): 0 for k in range(1, solution.K + 1) for t in range(1, solution.T + 1)}
    overtimes = {(k, t): 0 for k in range(1, solution.K + 1) for t in range(1, solution.T + 1)}
    normalized_overtimes = {}

    for (i, k, t) in solution.x.keys():
        utilizations[(k, t)] += actual_times[i]
        if actual_times[i] < solution.p[i]: # anticipo
            overtimes[(k, t)] += solution.p[i]
        else: # ritardo
            overtimes[(k, t)] += actual_times[i]

    for (k, t) in solution.s.keys():
        utilizations[(k, t)] = utilizations[(k, t)] / solution.s[(k, t)]
        normalized_overtimes[(k, t)] = overtimes[(k, t)] / solution.s[(k, t)]

    return {"U": utilizations, "O": overtimes, "Norm_O": normalized_overtimes}

utilization_and_overtime = compute_real_world_utilization_and_overtime(planner.solution, dl.actual_operating_times)
print(utilization_and_overtime)

selected_patients = extract_selected_patients_dict(planner.solution, solver_input)
pprint(selected_patients)

selected_patients_ids = extract_patient_ids(selected_patients)
pprint(selected_patients_ids)

# id_lists: list of lists where each list contains patients having same precedence (in increasing order)
# returns the list of permutations of the input list (considered as the concatenation of lists in id_lists)
def compute_ids_permutations(id_lists):
    iterators = [itertools.permutations(id_list) for id_list in id_lists]
    product_iterator = itertools.product(*iterators)
    return [list(itertools.chain(*permutation)) for permutation in product_iterator]

pprint(compute_best_permutations(selected_patients))

pprint(date_t_map)