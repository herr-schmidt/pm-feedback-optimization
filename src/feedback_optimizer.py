from data_loader import DataLoader
from planners import HeuristicLBBDPlanner
from utils import SolutionVisualizer
from datetime import datetime, timedelta
from pandas import DataFrame, read_excel, ExcelWriter
# from Predict_DAY_delay_class import day_delay_predictor
from Predict_DAY_delay_class_senza_pm1 import day_delay_predictor
import bisect
import itertools
import random

class MLFeedbackOptimizer():

    def __init__(self):
        self.planner = HeuristicLBBDPlanner(timeLimit=60, gap = 0.000, iterations_cap=30, solver="cplex")
        self.data_loader = DataLoader()
        self.solver_input = None
        self.solution_visualizer = SolutionVisualizer()

        self.phase_2_predictor = day_delay_predictor()

        self.threshold = 305
        self.date_t_map = {}
        self.overall_solution = {}
        self.fixed_ORs = {}

    def create_model(self, predict_operating_times=True, input_filepath="../input/input.xlsx"):
        self.data_loader.load_input_file(input_filepath, predicted_operating_times=predict_operating_times)

        self.solver_input = self.data_loader.generate_solver_input()
        self.planner.create_model(self.solver_input)

    def solve_model(self):
        self.planner.solve_model(self.solver_input)

    def visualize_solution(self):
        # sv.print_solution(solution)
        print("Planned patients:" + str(self.solution_visualizer.count_operated_patients(self.planner.solution)))
        self.solution_visualizer.plot_graph(self.planner.solution)

    # compute utilization by pretending the computed schedule was succesfully implemented for
    # all planned patients, but with the real-world time. WARNING: this cannot be done if more
    # than 1 operating room is used AND anesthesiae are taken into account.
    def compute_real_world_utilization_and_overtime(self):
        utilizations = {(k, t): 0 for k in range(1, self.planner.solution.K + 1) for t in range(1, self.planner.solution.T + 1)}
        overtimes = {(k, t): 0 for k in range(1, self.planner.solution.K + 1) for t in range(1, self.planner.solution.T + 1)}
        normalized_overtimes = {}

        for (i, k, t) in self.planner.solution.x.keys():
            utilizations[(k, t)] += self.data_loader.actual_operating_times[i]
            if self.data_loader.actual_operating_times[i] < self.planner.solution.p[i]: # anticipo
                overtimes[(k, t)] += self.planner.solution.p[i]
            else: # ritardo
                overtimes[(k, t)] += self.data_loader.actual_operating_times[i]

        for (k, t) in self.planner.solution.s.keys():
            util_denominator = self.planner.solution.s[(k, t)]
            if overtimes[(k, t)] > self.planner.solution.s[(k, t)]:
                util_denominator = overtimes[(k, t)]
            utilizations[(k, t)] = utilizations[(k, t)] / util_denominator # self.planner.solution.s[(k, t)]
            normalized_overtimes[(k, t)] = overtimes[(k, t)] / self.planner.solution.s[(k, t)]

        return {"U": utilizations, "O": overtimes, "Norm_O": normalized_overtimes}

    # extracts a python dict of the form {date: list((gamma, patient_id, precedence))}
    # needed for the PM2 phase
    def extract_selected_patients_dict(self):
        starting_day = datetime(year=2022, month=9, day=5)

        selected_patients = {}

        for (i, k, t) in self.planner.solution.x.keys():
            offset_weeks = (t - 1) // 5
            offset_days =  (t - 1) % 5
            day = starting_day + timedelta(days=offset_days, weeks=offset_weeks) # t - 1 since solver has t = 1 for the first day, and we want to include starting day
            patient_id = int(self.solver_input[None]["patientId"][i])

            gamma_id_precedence = (self.planner.solution.gamma[i], patient_id, self.planner.solution.precedence[i])

            if day in selected_patients:
                bisect.insort(selected_patients[day], gamma_id_precedence)
            else:
                selected_patients[day] = [gamma_id_precedence]

            # in order to easily retrieve t's when reaccessing solver's variables
            self.date_t_map[day] = t

        self.selected_patients = selected_patients

    def extract_patient_ids(self):
        selected_patients = {}

        for date in self.selected_patients.keys():
            selected_patients[date] = list(map(lambda gamma_id_precedence: gamma_id_precedence[1], self.selected_patients[date]))

        return selected_patients

    # PM2
    # return evaluation
    def evaluate_slot(self, date, patients_ids):
        return self.phase_2_predictor.get_day_prediction({date: patients_ids})[date]

    # compute, for each non-fixed slot, the permutations on which PM2 achieves the minimum overtime
    def compute_best_permutations(self):
        # will contain the tuple (permutation, permutation_evaluation) for each date
        best_permutations = {}
        
        for date in self.selected_patients.keys():
            # if slot was already ok, do not consider it
            # if self.fixed_ORs[(1, self.date_t_map[date])] == 1:
            #     continue
            # extract a list of lists, according to the prerequisites of compute_ids_permutations (namely, a list of lists each one of which is
            # relative to a different precedence)
            current_precedence = 1
            current_precedence_partition = []
            id_partition_by_precedence = []
            for (_, id, precedence) in self.selected_patients[date]:
                if precedence == current_precedence:
                    current_precedence_partition.append(id)
                else:
                    id_partition_by_precedence.append(current_precedence_partition)
                    current_precedence = precedence
                    current_precedence_partition = [id]

            # append last chunk
            id_partition_by_precedence.append(current_precedence_partition)

            permutations = self.compute_ids_permutations(id_partition_by_precedence)
            evaluations = [(permutation, self.evaluate_slot(date, permutation)) for permutation in permutations]
            best_permutation = min(evaluations, key=lambda e:e[1])

            best_permutations[date] = best_permutation

        return best_permutations

    # id_lists: list of lists where each list contains patients having same precedence (in increasing order)
    # returns the list of permutations of the input list (considered as the concatenation of lists in id_lists)
    def compute_ids_permutations(self, id_lists):
        iterators = [itertools.permutations(id_list) for id_list in id_lists]
        product_iterator = itertools.product(*iterators)
        return random.choices([list(itertools.chain(*permutation)) for permutation in product_iterator], k=100)

    def fix_OR(self, k, t):
        self.planner.solver_input[None]["s"][(k, t)] = 0

    # fix slots which are considered safe (within threshold) by PM2
    def check_overtime(self):
        for date in self.selected_patients:
            patients_ids = list(map(lambda gamma_id_precedence: gamma_id_precedence[1], self.selected_patients[date]))
            evaluation = self.evaluate_slot(date, patients_ids)
            print(evaluation)
            if evaluation <= self.threshold:
                self.overall_solution[date] = self.selected_patients[date]
                self.fixed_ORs[(1, self.date_t_map[date])] = 1
            else:
                self.fixed_ORs[(1, self.date_t_map[date])] = 0

    def fix_model_ORs(self):
        for (i, k, t) in self.planner.solution.x:
            if self.fixed_ORs[(k, t)] == 1:
                self.planner.MP_instance.x[i, k, t].fix(1)


    def adjust_with_permutations(self):
        best_permutations = self.compute_best_permutations() # tuples: 0 contains permutation, 1 contains evaluation
        for date in best_permutations:
            if best_permutations[date][1] <= self.threshold:
                self.overall_solution[date] = self.best_permutations[date][0]
                self.fixed_ORs[(1, self.date_t_map[date])] = 1

    # new input file for re-launching optimization. Shall contain only non-assigned patients
    def generate_post_optimization_input(self):
        phase_1_input_dataframe = read_excel(io="../input/input.xlsx", sheet_name="patients")
        assigned_patients = []
        for selected_patients_list in self.overall_solution.values():
            assigned_patients.extend(selected_patients_list)

        post_opt_input_dataframe = phase_1_input_dataframe[phase_1_input_dataframe["Id"] not in assigned_patients]
        
        with ExcelWriter(path="input/post_opt.xlsx", mode="w", engine="openpyxl") as writer:
            post_opt_input_dataframe.to_excel(excel_writer=writer, index=False, header=True)