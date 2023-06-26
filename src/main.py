from data_loader import DataLoader
from planners import HeuristicLBBDPlanner
from utils import SolutionVisualizer
from datetime import datetime, timedelta
from pprint import pprint

# extracts a python dict of the form {date: list(patient_ids)}
# needed for the PM2 phase
def extract_selected_patients_dict(solution, solver_input):
    starting_day = datetime(year=2022, month=9, day=5)

    selected_patients = {}

    for (i, k, t) in solution.x.keys():
        offset_weeks = (t - 1) // 5
        offset_days =  (t - 1) % 5
        day = starting_day + timedelta(days=offset_days, weeks=offset_weeks) # t - 1 since solver has t = 1 for the first day, and we want to include starting day
        patient_id = int(solver_input[None]["patientId"][i])
        if day in selected_patients:
            selected_patients[day].append(patient_id)
        else:
            selected_patients[day] = [patient_id]

    return selected_patients

dl = DataLoader()
dl.load_input_file("../input/input.xlsx", predicted_operating_times=True)

solver_input = dl.generate_solver_input()
print(solver_input)

planner = HeuristicLBBDPlanner(timeLimit=290, gap = 0.005, iterations_cap=30, solver="cplex")
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
