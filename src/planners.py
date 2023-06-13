from __future__ import division
import re
import time
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from math import isclose, inf

from abc import ABC, abstractmethod

from planner.model import Patient


class Planner(ABC):
    
    def __init__(self, timeLimit, gap, solver):
        self.solver = pyo.SolverFactory(solver)
        if(solver == "cplex"):
            self.timeLimit = 'timelimit'
            # self.gap = 'mip tolerances mipgap'
            self.gap = 'mip tolerances mipgap'
            self.solver.options['emphasis'] = "mip 3"
            # self.solver.options['threads'] = 6
        if(solver == "gurobi"):
            self.timeLimit = 'timelimit'
            self.gap = 'mipgap'
            self.solver.options['mipfocus'] = 2
        if(solver == "cbc"):
            self.timeLimit = 'seconds'
            self.gap = 'ratiogap'
            self.solver.options['heuristics'] = "on"
            # self.solver.options['round'] = "on"
            # self.solver.options['feas'] = "on"
            self.solver.options['cuts'] = "on"
            self.solver.options['preprocess'] = "on"
            # self.solver.options['printingOptions'] = "normal"

        self.solver.options[self.timeLimit] = timeLimit
        self.solver.options[self.gap] = gap

        self.reset_run_info()

    def reset_run_info(self):
        self.solver_time = 0
        self.cumulated_building_time = 0
        self.status_ok = False
        self.gap = 0
        self.MP_objective_function_value = 0
        self.objective_function_value = 0
        self.MP_time_limit_hit = False
        self.time_limit_hit = False
        self.MP_upper_bound = 0
        self.upper_bound = 0
        self.generated_constraints = 0
        self.discarded_constraints = 0


    @abstractmethod
    def extract_run_info(self):
        pass

    @abstractmethod
    def define_model(self):
        pass

    def single_surgery_rule(self, model, i):
        self.generated_constraints += 1
        return sum(model.x[i, k, t] for k in model.k for t in model.t) <= 1

    def single_delay_rule(self, model, i):
        self.generated_constraints += 1
        return sum(model.delta[q, i, k, t] for k in model.k for t in model.t for q in model.q) <= 1

    def robustness_constraints_rule(self, model, q, k, t):
        self.generated_constraints += 1
        return sum(model.delta[q, i, k, t] for i in model.i) <= model.Gamma[q, k, t]

    def delay_implication_constraint_rule(self, model, i, k, t):
        self.generated_constraints += 1
        return model.x[i, k, t] >= sum(model.delta[q, i, k, t] for q in model.q)

    def surgery_time_rule(self, model, k, t):
        self.generated_constraints += 1
        return sum(model.p[i] * model.x[i, k, t] + sum(model.d[q, i] * model.delta[q, i, k, t] for q in model.q) for i in model.i) <= model.s[k, t]

    def specialty_assignment_rule(self, model, j, k, t):
        self.generated_constraints += 1
        return sum(model.x[i, k, t] for i in model.i if model.specialty[i] == j) <= model.bigM[1] * model.tau[j, k, t]

    def anesthetist_assignment_rule(self, model, i, t):
        self.generated_constraints += 1
        return sum(model.beta[alpha, i, t] for alpha in model.alpha) == model.a[i] * sum(model.x[i, k, t] for k in model.k)

    def anesthetist_time_rule(self, model, alpha, t):
        if sum(model.a[i] for i in model.i) == 0:
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return sum(model.beta[alpha, i, t] * model.p[i] for i in model.i if model.a[i] == 1) + sum(model.z[q, alpha, i, k, t] * model. d[q, i] for i in model.i for k in model.k for q in model.q if model.a[i] == 1) <= model.An[alpha, t]

    # needed for linearizing product of binary variables
    def z_rule_1(self, model, q, alpha, i, k, t):
        self.generated_constraints += 1
        return model.z[q, alpha, i, k, t] <= model.beta[alpha, i, t]

    def z_rule_2(self, model, q, alpha, i, k, t):
        self.generated_constraints += 1
        return model.z[q, alpha, i, k, t] <= model.delta[q, i, k, t]

    def z_rule_3(self, model, q, alpha, i, k, t):
        self.generated_constraints += 1
        return model.z[q, alpha, i, k, t] >= model.beta[alpha, i, t] + model.delta[q, i, k, t] - 1
    
    def symmetry_rule(self, model, t1, t2):
        if t1 >= t2:
            return pyo.Constraint.Skip
        return sum(model.x[i, k, t1] for i in model.i for k in model.k) >= sum(model.x[i, k, t2] for i in model.i for k in model.k)

    # patients with same anesthetist on same day but different room cannot overlap
    @abstractmethod
    def anesthetist_no_overlap_rule(self, model, i1, i2, k1, k2, t, alpha):
        pass

    # precedence across rooms
    @abstractmethod
    def lambda_rule(self, model, i1, i2, t):
        pass

    # ensure gamma plus operation time does not exceed end of day
    @abstractmethod
    def end_of_day_rule(self, model, i, k, t):
        pass

    # ensure that patient i1 terminates operation before i2, if y_12kt = 1
    @abstractmethod
    def time_ordering_precedence_rule(self, model, i1, i2, k, t):
        pass

    @abstractmethod
    def start_time_ordering_priority_rule(self, model, i1, i2, k, t):
        pass

    # either i1 comes before i2 in (k, t) or i2 comes before i1 in (k, t)
    @abstractmethod
    def exclusive_precedence_rule(self, model, i1, i2, k, t):
        pass

    def objective_function(self, model):
        N = (sum(model.r[i] for i in model.i))
        R = sum(model.x[i, k, t] * model.r[i] for i in model.i for k in model.k for t in model.t)
        D = sum(model.d[q, i] * model.delta[q, i, k, t] for i in model.i for k in model.k for t in model.t for q in model.q)
        return  D + R / N

    # constraints
    def define_single_surgery_constraints(self, model):
        model.single_surgery_constraint = pyo.Constraint(
            model.i,
            rule=lambda model, i: self.single_surgery_rule(model, i))

    def define_single_delay_constraints(self, model):
        model.single_surgery_delay_constraint = pyo.Constraint(
            model.i,
            rule=lambda model, i: self.single_delay_rule(model, i))

    def define_robustness_constraints(self, model):
        model.robustness_constraint = pyo.Constraint(
            model.q,
            model.k,
            model.t,
            rule=lambda model, q, k, t: self.robustness_constraints_rule(model, q, k, t))

    def define_delay_implication_constraint(self, model):
        model.delay_implication_constraint = pyo.Constraint(
            model.i,
            model.k,
            model.t,
            rule=lambda model, i, k, t: self.delay_implication_constraint_rule(model, i, k, t))

    def define_surgery_time_constraints(self, model):
        model.surgery_time_constraint = pyo.Constraint(
            model.k,
            model.t,
            rule=lambda model, k, t: self.surgery_time_rule(model, k, t))

    def define_specialty_assignment_constraints(self, model):
        model.specialty_assignment_constraint = pyo.Constraint(
            model.j,
            model.k,
            model.t,
            rule=lambda model, j, k, t: self.specialty_assignment_rule(model, j, k, t))

    def define_anesthetist_assignment_constraint(self, model):
        model.anesthetist_assignment_constraint = pyo.Constraint(
            model.i,
            model.t,
            rule=lambda model, i, t: self.anesthetist_assignment_rule(model, i, t))

    def define_anesthetist_time_constraint(self, model):
        model.anesthetist_time_constraint = pyo.Constraint(
            model.alpha,
            model.t,
            rule=lambda model, alpha, t: self.anesthetist_time_rule(model, alpha, t))

    def define_anesthetist_no_overlap_constraint(self, model):
        model.anesthetist_no_overlap_constraint = pyo.Constraint(
            model.i,
            model.i,
            model.k,
            model.k,
            model.t,
            model.alpha,
            rule=lambda model, i1, i2, k1, k2, t, alpha: self.anesthetist_no_overlap_rule(model, i1, i2, k1, k2, t, alpha))

    def define_lambda_constraint(self, model):
        model.lambda_constraint = pyo.Constraint(
            model.i,
            model.i,
            model.t,
            rule=lambda model, i1, i2, t: self.lambda_rule(model, i1, i2, t))

    def define_end_of_day_constraint(self, model):
        model.end_of_day_constraint = pyo.Constraint(
            model.i,
            model.k,
            model.t,
            rule=lambda model, i, k, t: self.end_of_day_rule(model, i, k, t))

    def define_priority_constraint(self, model):
        model.priority_constraint = pyo.Constraint(
            model.i,
            model.i,
            model.k,
            model.t,
            rule=lambda model, i1, i2, k, t: self.start_time_ordering_priority_rule(model, i1, i2, k, t))

    def define_precedence_constraint(self, model):
        model.precedence_constraint = pyo.Constraint(
            model.i,
            model.i,
            model.k,
            model.t,
            rule=lambda model, i1, i2, k, t: self.time_ordering_precedence_rule(model, i1, i2, k, t))

    def define_exclusive_precedence_constraint(self, model):
        model.exclusive_precedence_constraint = pyo.Constraint(
            model.i,
            model.i,
            model.k,
            model.t,
            rule=lambda model, i1, i2, k, t: self.exclusive_precedence_rule(model, i1, i2, k, t))

    def define_z_constraints(self, model):
        model.z_constraints_1 = pyo.Constraint(
            model.q,
            model.alpha,
            model.i,
            model.k,
            model.t,
            rule=lambda model, q, alpha, i, k, t: self.z_rule_1(model, q, alpha, i, k, t))

        model.z_constraints_2 = pyo.Constraint(
            model.q,
            model.alpha,
            model.i,
            model.k,
            model.t,
            rule=lambda model, q, alpha, i, k, t: self.z_rule_2(model, q, alpha, i, k, t))

        model.z_constraints_3 = pyo.Constraint(
            model.q,
            model.alpha,
            model.i,
            model.k,
            model.t,
            rule=lambda model, q, alpha, i, k, t: self.z_rule_3(model, q, alpha, i, k, t))
        
    def define_symmetry_constraints(self, model):
            model.symmetry_constraint = pyo.Constraint(
            model.t,
            model.t,
            rule=lambda model, t1, t2: self.symmetry_rule(model, t1, t2))

    def define_objective(self, model):
        model.objective = pyo.Objective(
            rule=self.objective_function,
            sense=pyo.maximize)

    def define_lambda_variables(self, model):
        model.Lambda = pyo.Var(model.i,
                               model.i,
                               model.t,
                               domain=pyo.Binary)

    def define_y_variables(self, model):
        model.y = pyo.Var(model.i,
                          model.i,
                          model.k,
                          model.t,
                          domain=pyo.Binary)

    def define_gamma_variables(self, model):
        model.gamma = pyo.Var(model.i, domain=pyo.NonNegativeReals)

    def define_anesthetists_number_param(self, model):
        model.A = pyo.Param(within=pyo.NonNegativeIntegers)

    def define_anesthetists_range_set(self, model):
        model.alpha = pyo.RangeSet(1, model.A)

    def define_beta_variables(self, model):
        model.beta = pyo.Var(model.alpha,
                             model.i,
                             model.t,
                             domain=pyo.Binary)

    def define_anesthetists_availability(self, model):
        model.An = pyo.Param(model.alpha, model.t)

    def define_sets(self, model):
        model.I = pyo.Param(within=pyo.NonNegativeIntegers)
        model.J = pyo.Param(within=pyo.NonNegativeIntegers)
        model.K = pyo.Param(within=pyo.NonNegativeIntegers)
        model.T = pyo.Param(within=pyo.NonNegativeIntegers)
        model.M = pyo.Param(within=pyo.NonNegativeIntegers)
        model.Q = pyo.Param(within=pyo.NonNegativeIntegers)

        model.i = pyo.RangeSet(1, model.I)
        model.j = pyo.RangeSet(1, model.J)
        model.k = pyo.RangeSet(1, model.K)
        model.t = pyo.RangeSet(1, model.T)
        model.bigMRangeSet = pyo.RangeSet(1, model.M)
        model.q = pyo.RangeSet(1, model.Q)

    def define_x_variables(self, model):
        model.x = pyo.Var(model.i,
                          model.k,
                          model.t,
                          domain=pyo.Binary)

    def define_delta_variables(self, model):
        model.delta = pyo.Var(model.q,
                              model.i,
                              model.k,
                              model.t,
                              domain=pyo.Binary)

    def define_z_variables(self, model):
        model.z = pyo.Var(model.q,
                          model.alpha,
                          model.i,
                          model.k,
                          model.t,
                          domain=pyo.Binary)

    def define_parameters(self, model):
        model.p = pyo.Param(model.i)
        model.d = pyo.Param(model.q, model.i)
        model.r = pyo.Param(model.i)
        model.s = pyo.Param(model.k, model.t)
        model.a = pyo.Param(model.i)
        model.c = pyo.Param(model.i)
        model.u = pyo.Param(model.i, model.i)
        model.tau = pyo.Param(model.j, model.k, model.t)
        model.specialty = pyo.Param(model.i)
        model.bigM = pyo.Param(model.bigMRangeSet)
        model.precedence = pyo.Param(model.i)
        model.Gamma = pyo.Param(model.q, model.k, model.t)

    def extract_solution(self):
        if self.solution:
            return self.solution.to_patients_dict()

        return None

    def compute_specialty_selection_ratio(self):
        specialty_selection_ratio = None
        if self.solution:
            specialty_selection_ratio = {j: 0 for j in range(1, self.solution.J + 1)}
            for j in range(1, self.solution.J + 1):
                specialty_j_selected_patients = 0
                specialty_j_total_patients = 0
                for (i, _, _) in self.solution.x:
                    if self.solution.specialty[i] == j:
                        specialty_j_selected_patients += 1
                for i in range(1, self.solution.I + 1):
                    if self.solution.specialty[i] == j:
                        specialty_j_total_patients += 1
                specialty_selection_ratio[j] = specialty_j_selected_patients / specialty_j_total_patients

        return specialty_selection_ratio
    
    def compute_operating_room_utilization_by_specialty(self):
        operating_room_utilization = self.compute_operating_room_utilization()
        OR_utilization_by_specialty = None
        if operating_room_utilization:
            OR_utilization_by_specialty = {j: 0 for j in range(1, self.solution.J + 1)}
            for j in range(1, self.solution.J + 1):
                specialty_j_ORs = 0
                cumulated_utilization = 0
                for (k, t) in operating_room_utilization.keys():
                    if self.solution.tau[(j, k, t)] == 1:
                        specialty_j_ORs += 1
                        cumulated_utilization += operating_room_utilization[(k, t)]
                OR_utilization_by_specialty[j] = cumulated_utilization / specialty_j_ORs

        return OR_utilization_by_specialty
            
    def compute_operating_room_utilization(self):
        room_utilization = None
        if self.solution:
            room_utilization = {(k, t): 0 for (_, k, t) in self.solution.x}
            for (i, k, t) in self.solution.x:
                room_utilization[(k, t)] += self.solution.p[i] + sum(self.solution.d[q, i] for (q, i1, _, _) in self.solution.delta if i == i1)
            for (k, t) in room_utilization.keys():
                room_utilization[(k, t)] = room_utilization[(k, t)] / self.solution.s[(k, t)]
        return room_utilization


class SimplePlanner(Planner):

    def __init__(self, timeLimit, gap, solver):
        super().__init__(timeLimit, gap, solver)
        self.model = pyo.AbstractModel()
        self.model_instance = None

    def anesthetist_no_overlap_rule(self, model, i1, i2, k1, k2, t, alpha):
        if(i1 == i2 or k1 == k2 or model.a[i1] * model.a[i2] == 0):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.gamma[i1] + model.p[i1] + sum(model.d[q, i1] * model.delta[q, i1, k1, t] for q in model.q) <= model.gamma[i2] + model.bigM[2] * (5 - model.beta[alpha, i1, t] - model.beta[alpha, i2, t] - model.x[i1, k1, t] - model.x[i2, k2, t] - model.Lambda[i1, i2, t])

    def lambda_rule(self, model, i1, i2, t):
        if(i1 >= i2 or not (model.a[i1] == 1 and model.a[i2] == 1)):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.Lambda[i1, i2, t] + model.Lambda[i2, i1, t] == 1

    def end_of_day_rule(self, model, i, k, t):
        if(model.tau[model.specialty[i], k, t] == 0):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.gamma[i] + model.p[i] + sum(model.d[q, i] * model.delta[q, i, k, t] for q in model.q) <= model.s[k, t]

    def time_ordering_precedence_rule(self, model, i1, i2, k, t):
        if(i1 == i2
           or (model.specialty[i1] != model.specialty[i2])
           or (model.tau[model.specialty[i1], k, t] == 0)):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.gamma[i1] + model.p[i1] + sum(model.d[q, i1] * model.delta[q, i1, k, t] for q in model.q) <= model.gamma[i2] + model.bigM[2] * (3 - model.x[i1, k, t] - model.x[i2, k, t] - model.y[i1, i2, k, t])

    def start_time_ordering_priority_rule(self, model, i1, i2, k, t):
        if(i1 == i2 or model.u[i1, i2] == 0
           or (model.specialty[i1] != model.specialty[i2])
           or (model.tau[model.specialty[i1], k, t] == 0)):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.gamma[i1] * model.u[i1, i2] <= model.gamma[i2] * (1 - model.u[i2, i1]) + model.bigM[2] * (2 - model.x[i1, k, t] - model.x[i2, k, t])

    def exclusive_precedence_rule(self, model, i1, i2, k, t):
        if(i1 >= i2
           or (model.specialty[i1] != model.specialty[i2])
           or (model.tau[model.specialty[i1], k, t] == 0)):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.y[i1, i2, k, t] + model.y[i2, i1, k, t] == 1

    def define_model(self):
        self.define_sets(self.model)
        self.define_parameters(self.model)
        self.define_x_variables(self.model)
        self.define_delta_variables(self.model)
        self.define_single_delay_constraints(self.model)
        self.define_robustness_constraints(self.model)
        self.define_delay_implication_constraint(self.model)

        self.define_single_surgery_constraints(self.model)
        self.define_surgery_time_constraints(self.model)
        self.define_specialty_assignment_constraints(self.model)

        self.define_anesthetists_number_param(self.model)
        self.define_anesthetists_range_set(self.model)
        self.define_beta_variables(self.model)
        self.define_anesthetists_availability(self.model)
        self.define_lambda_variables(self.model)
        self.define_y_variables(self.model)
        self.define_gamma_variables(self.model)

        self.define_anesthetist_assignment_constraint(self.model)
        self.define_z_variables(self.model)
        self.define_z_constraints(self.model)
        self.define_symmetry_constraints(self.model)
        self.define_anesthetist_time_constraint(self.model)
        self.define_anesthetist_no_overlap_constraint(self.model)
        self.define_lambda_constraint(self.model)
        self.define_end_of_day_constraint(self.model)
        self.define_priority_constraint(self.model)
        self.define_precedence_constraint(self.model)
        self.define_exclusive_precedence_constraint(self.model)

        self.define_objective(self.model)

    def create_model_instance(self, data):
        print("Creating model instance...")
        t = time.time()
        self.model_instance = self.model.create_instance(data)
        elapsed = (time.time() - t)
        self.cumulated_building_time += elapsed

    def fix_vars(self, model_instance):
        for i in model_instance.i:
            if model_instance.a[i] == 0:
                for alpha in model_instance.alpha:
                    for t in model_instance.t:
                        model_instance.beta[alpha, i, t].fix(0)
            for k in model_instance.k:
                for t in model_instance.t:
                    if model_instance.tau[model_instance.specialty[i], k, t] == 0:
                        model_instance.x[i, k, t].fix(0)
                        model_instance.delta[1, i, k, t].fix(0)
                        for alpha in model_instance.alpha:
                            model_instance.z[1, alpha, i, k, t].fix(0)


    def fix_y_variables(self, model_instance):
        print("Fixing y variables...")
        fixed = 0
        for k in model_instance.k:
            for t in model_instance.t:
                for i1 in range(2, self.model_instance.I + 1):
                    for i2 in range(1, i1):
                        if(model_instance.u[i1, i2] == 1):
                            model_instance.y[i1, i2, k, t].fix(1)
                            model_instance.y[i2, i1, k, t].fix(0)
                            fixed += 2
        print(str(fixed) + " y variables fixed.")

    def extract_run_info(self):
        OR_utilization_by_specialty = self.compute_operating_room_utilization_by_specialty()
        specialty_selection_ratio = self.compute_specialty_selection_ratio()

        specialty_1_OR_utilization = None
        specialty_2_OR_utilization = None
        specialty_1_selection_ratio = None
        specialty_2_selection_ratio = None

        if OR_utilization_by_specialty:
            specialty_1_OR_utilization = OR_utilization_by_specialty[1]
            specialty_2_OR_utilization = OR_utilization_by_specialty[2]
        if specialty_selection_ratio:
            specialty_1_selection_ratio = specialty_selection_ratio[1]
            specialty_2_selection_ratio = specialty_selection_ratio[2]

        return {"cumulated_building_time": self.cumulated_building_time,
                "solver_time": self.solver_time,
                "time_limit_hit": self.time_limit_hit,
                "upper_bound": self.upper_bound,
                "status_ok": self.status_ok,
                "gap": self.gap,
                "objective_function_value": self.solution.objective_value,
                "specialty_1_OR_utilization": specialty_1_OR_utilization,
                "specialty_2_OR_utilization": specialty_2_OR_utilization,
                "specialty_1_selection_ratio": specialty_1_selection_ratio,
                "specialty_2_selection_ratio": specialty_2_selection_ratio,
                "generated_constraints": self.generated_constraints,
                "discarded_constraints": self.discarded_constraints,
                "discarded_constraints_ratio": self.discarded_constraints / (self.discarded_constraints + self.generated_constraints)
                }

    def solve_model(self, data):
        self.reset_run_info()
        self.define_model()
        self.create_model_instance(data)
        self.fix_vars(self.model_instance)
        self.fix_y_variables(self.model_instance)
        print("Solving model instance...")
        self.model.results = self.solver.solve(self.model_instance, tee=True)
        print("\nModel instance solved.")
        self.solver_time = self.solver._last_solve_time
        resultsAsString = str(self.model.results)
        self.upper_bound = float(re.search("Upper bound: -*(\d*\.\d*)", resultsAsString).group(1))
        self.gap = round((1 - pyo.value(self.model_instance.objective) / self.upper_bound) * 100, 2)

        self.time_limit_hit = self.model.results.solver.termination_condition in [TerminationCondition.maxTimeLimit]
        self.status_ok = self.model.results.solver.status == SolverStatus.ok

        self.solution = Solution(self.model_instance)


class TwoPhasePlanner(Planner):

    def __init__(self, timeLimit, gap, solver):
        super().__init__(timeLimit, gap, solver)
        self.MP_model = pyo.AbstractModel()
        self.MP_instance = None
        self.SP_model = pyo.AbstractModel()
        self.SP_instance = None

    def define_model(self):
        self.define_MP()
        self.define_SP()

        self.define_objective(self.MP_model)
        self.define_objective(self.SP_model)

    def define_MP(self):
        self.define_sets(self.MP_model)
        self.define_parameters(self.MP_model)
        self.define_x_variables(self.MP_model)
        self.define_delta_variables(self.MP_model)
        self.define_single_delay_constraints(self.MP_model)
        self.define_robustness_constraints(self.MP_model)
        self.define_delay_implication_constraint(self.MP_model)
        self.define_single_surgery_constraints(self.MP_model)
        self.define_surgery_time_constraints(self.MP_model)
        self.define_specialty_assignment_constraints(self.MP_model)
        self.define_anesthetists_number_param(self.MP_model)
        self.define_anesthetists_range_set(self.MP_model)
        self.define_beta_variables(self.MP_model)
        self.define_anesthetists_availability(self.MP_model)
        self.define_anesthetist_assignment_constraint(self.MP_model)
        self.define_z_variables(self.MP_model)
        self.define_z_constraints(self.MP_model)
        self.define_symmetry_constraints(self.MP_model)
        self.define_anesthetist_time_constraint(self.MP_model)

    def define_x_parameters(self):
        self.SP_model.x_param = pyo.Param(self.SP_model.i,
                                          self.SP_model.k,
                                          self.SP_model.t)

    def define_SP(self):
        self.define_sets(self.SP_model)
        self.define_parameters(self.SP_model)
        self.define_x_variables(self.SP_model)
        self.define_delta_variables(self.SP_model)
        self.define_single_delay_constraints(self.SP_model)
        self.define_robustness_constraints(self.SP_model)
        self.define_delay_implication_constraint(self.SP_model)
        self.define_single_surgery_constraints(self.SP_model)
        self.define_surgery_time_constraints(self.SP_model)
        self.define_specialty_assignment_constraints(self.SP_model)
        self.define_anesthetists_number_param(self.SP_model)
        self.define_anesthetists_range_set(self.SP_model)
        self.define_beta_variables(self.SP_model)
        self.define_anesthetists_availability(self.SP_model)
        self.define_anesthetist_assignment_constraint(self.SP_model)
        self.define_z_variables(self.SP_model)
        self.define_z_constraints(self.SP_model)
        self.define_symmetry_constraints(self.SP_model)
        self.define_anesthetist_time_constraint(self.SP_model)

        # SP's components
        self.define_x_parameters()
        self.define_lambda_variables(self.SP_model)
        self.define_y_variables(self.SP_model)
        self.define_gamma_variables(self.SP_model)
        self.define_anesthetist_no_overlap_constraint(self.SP_model)
        self.define_lambda_constraint(self.SP_model)
        self.define_end_of_day_constraint(self.SP_model)
        self.define_priority_constraint(self.SP_model)
        self.define_precedence_constraint(self.SP_model)
        self.define_exclusive_precedence_constraint(self.SP_model)

    def create_MP_instance(self, data):
        print("Creating MP instance...")
        t = time.time()
        self.MP_instance = self.MP_model.create_instance(data)
        elapsed = (time.time() - t)
        print("MP instance created in " + str(round(elapsed, 2)) + "s")
        self.cumulated_building_time += elapsed

    def create_SP_instance(self, data):
        self.extend_data(data)
        print("Creating SP instance...")
        t = time.time()
        self.SP_instance = self.SP_model.create_instance(data)
        elapsed = (time.time() - t)
        print("SP instance created in " + str(round(elapsed, 2)) + "s")
        self.cumulated_building_time += elapsed

    def solve_MP(self):
        print("Solving MP instance...")
        self.MP_model.results = self.solver.solve(self.MP_instance, tee=True)
        print("\nMP instance solved.")
        self.solver_time += self.solver._last_solve_time
        self.MP_time_limit_hit = self.MP_model.results.solver.termination_condition in [TerminationCondition.maxTimeLimit]
        self.MP_objective_function_value = pyo.value(self.MP_instance.objective)

        self.objective_values.append(self.MP_objective_function_value)

        resultsAsString = str(self.MP_model.results)
        self.MP_upper_bound = float(re.search("Upper bound: -*(\d*\.\d*)", resultsAsString).group(1))

    def solve_SP(self):
        print("Solving SP instance...")
        self.SP_model.results = self.solver.solve(self.SP_instance, tee=True)
        print("SP instance solved.")
        self.solver_time += self.solver._last_solve_time
        self.time_limit_hit = self.SP_model.results.solver.termination_condition in [
            TerminationCondition.maxTimeLimit]

class LBBDPlanner(TwoPhasePlanner):

    def __init__(self, timeLimit, gap, iterations_cap, solver):
        super().__init__(timeLimit, gap, solver)
        self.iterations_cap = iterations_cap

    @abstractmethod
    def extend_data(self, data):
        pass

    @abstractmethod
    def fix_SP_variables(self):
        pass

    def has_solution(self):
        return self.SP_model.results.solver.termination_condition in {TerminationCondition.feasible,
                                                                      TerminationCondition.optimal,
                                                                      TerminationCondition.maxTimeLimit
                                                                     }

    def solve_MP(self):
        super().solve_MP()
        residual_time = self.solver.options[self.timeLimit] - self.solver._last_solve_time
        if residual_time <= 0:
            self.last_round = True
            self.solver.options[self.timeLimit] = 10 # leave 10 seconds for solving the last SP
        else:
            self.solver.options[self.timeLimit] = residual_time

    def solve_SP(self):
        super().solve_SP()
        residual_time = self.solver.options[self.timeLimit] - self.solver._last_solve_time
        if residual_time <= 0:
            self.last_round = True
        else:
            self.solver.options[self.timeLimit] = residual_time

    def extract_run_info(self):
        OR_utilization_by_specialty = self.compute_operating_room_utilization_by_specialty()
        specialty_selection_ratio = self.compute_specialty_selection_ratio()

        specialty_1_OR_utilization = None
        specialty_2_OR_utilization = None
        specialty_1_selection_ratio = None
        specialty_2_selection_ratio = None

        if OR_utilization_by_specialty:
            specialty_1_OR_utilization = OR_utilization_by_specialty[1]
            specialty_2_OR_utilization = OR_utilization_by_specialty[2]
        if specialty_selection_ratio:
            specialty_1_selection_ratio = specialty_selection_ratio[1]
            specialty_2_selection_ratio = specialty_selection_ratio[2]

        return {"cumulated_building_time": self.cumulated_building_time,
                "solver_time": self.solver_time,
                "time_limit_hit": self.time_limit_hit,
                "upper_bound": self.upper_bound,
                "status_ok": self.status_ok,
                "gap": self.gap,
                "MP_objective_function_value": self.MP_objective_function_value,
                "objective_function_value": self.objective_function_value,
                "MP_upper_bound": self.MP_upper_bound,
                "MP_time_limit_hit": self.MP_time_limit_hit,
                "time_limit_hit": self.time_limit_hit,
                "iterations": self.iterations,
                "specialty_1_OR_utilization": specialty_1_OR_utilization,
                "specialty_2_OR_utilization": specialty_2_OR_utilization,
                "specialty_1_selection_ratio": specialty_1_selection_ratio,
                "specialty_2_selection_ratio": specialty_2_selection_ratio,
                "generated_constraints": self.generated_constraints,
                "discarded_constraints": self.discarded_constraints,
                "discarded_constraints_ratio": self.discarded_constraints / (self.discarded_constraints + self.generated_constraints)
                }

    def is_optimal(self):
        return isclose(pyo.value(self.MP_instance.objective), pyo.value(self.SP_instance.objective))

    def MP_anesthetist_time_rule(self, model, t):
        self.generated_constraints += 1
        return sum(model.a[i] * model.p[i] * model.x[i, k, t] for i in model.i for k in model.k) + sum(model.a[i] * model.d[q, i] * model.delta[q, i, k, t] for i in model.i for k in model.k for q in model.q) <= sum(model.An[alpha, t] for alpha in model.alpha)

    def define_MP_anesthetist_time_constraint(self, model):
        model.MP_anesthetist_time_constraint = pyo.Constraint(
            model.t,
            rule=lambda model, t: self.MP_anesthetist_time_rule(model, t))

    def add_objective_cut(self):
        # self.MP_instance.objective_function_cuts.clear()

        N = (sum(self.MP_instance.r[i] for i in self.MP_instance.i))
        R = sum(self.MP_instance.x[i, k, t] * self.MP_instance.r[i] for i in self.MP_instance.i for k in self.MP_instance.k for t in self.MP_instance.t)
        D = sum(self.MP_instance.d[q, i] * self.MP_instance.delta[q, i, k, t] for i in self.MP_instance.i for k in self.MP_instance.k for t in self.MP_instance.t for q in self.MP_instance.q)
        M = sum(self.MP_instance.d[q, i] for i in self.MP_instance.i for q in self.MP_instance.q)        
        cut = D + R / N <= pyo.value(self.MP_instance.objective)

        self.MP_instance.objective_function_cuts.add(cut)

    def add_patients_cut(self):
        self.MP_instance.patients_cuts.add(sum(
            1 - self.MP_instance.x[i, k, t] for i in self.MP_instance.i for k in self.MP_instance.k for t in self.MP_instance.t if round(self.MP_instance.x[i, k, t].value) == 1) >= 1)

        self.MP_instance.patients_cuts.display()

    def save_best_solution(self):
        SP_objective_value = pyo.value(self.SP_instance.objective)
        if SP_objective_value > self.best_SP_solution_value:
            self.best_SP_solution_value = SP_objective_value
            self.solution = Solution(self.SP_instance)

    def fix_MP_vars(self):
        for i in self.MP_instance.i:
            if self.MP_instance.a[i] == 0:
                for alpha in self.MP_instance.alpha:
                    for t in self.MP_instance.t:
                        self.MP_instance.beta[alpha, i, t].fix(0)
            if self.MP_instance.specialty[i] == 1:
                for k in [3, 4]:
                    for t in self.MP_instance.t:
                        self.MP_instance.x[i, k, t].fix(0)
                        self.MP_instance.delta[1, i, k, t].fix(0)
                        for alpha in self.MP_instance.alpha:
                            self.MP_instance.z[1, alpha, i, k, t].fix(0)
            if self.MP_instance.specialty[i] == 2:
                for k in [1, 2]:
                    for t in self.MP_instance.t:
                        self.MP_instance.x[i, k, t].fix(0)
                        self.MP_instance.delta[1, i, k, t].fix(0)
                        for alpha in self.MP_instance.alpha:
                            self.MP_instance.z[1, alpha, i, k, t].fix(0)

    def solve_model(self, data):
        self.reset_run_info()
        self.define_model()
        self.create_MP_instance(data)
        self.MP_instance.patients_cuts = pyo.ConstraintList()
        self.MP_instance.objective_function_cuts = pyo.ConstraintList()
        self.selected_x_indices = set()

        self.iterations = 0
        self.last_round = False
        self.solution = None
        self.MP_least_upper_bound = inf
        self.best_SP_solution_value = 0

        self.objective_values = []
        self.D_ikt = []
        self.NR = []

        while self.iterations < self.iterations_cap:
            self.iterations += 1
            # MP
            self.fix_MP_vars()
            self.solve_MP()

            if self.MP_upper_bound < self.MP_least_upper_bound:
                self.MP_least_upper_bound = self.MP_upper_bound

            # SP
            self.create_SP_instance(data)
            self.fix_SP_variables()
            self.solve_SP()

            if self.has_solution():
                self.save_best_solution()

            if (not self.has_solution() or not self.is_optimal()) and not self.last_round:
                # depending on the variables' fixing rule, this cut assumes a different meaning
                # guaranteed_feasibility -> optimality cut
                # fix_all -> feasibility cut
                self.add_patients_cut()
                # this cut has no feasibility/optimality meaning: it simply helps in achieving faster computation times
                self.add_objective_cut()
            else:
                break

        self.status_ok = self.SP_model.results and self.SP_model.results.solver.status == SolverStatus.ok
        self.compute_gap_and_solution_value()
        print(self.objective_values)
        print(self.D_ikt)
        print(self.NR)

    def compute_gap_and_solution_value(self):
        self.objective_function_value = None
        self.gap = None

        if self.solution:
            self.objective_function_value = self.solution.objective_value
        else:
            return

        self.gap = round((1 - self.objective_function_value / self.MP_least_upper_bound) * 100, 6)


class HeuristicLBBDPlanner(LBBDPlanner):

    def anesthetist_no_overlap_rule(self, model, i1, i2, k1, k2, t, alpha):
        if(model.x_param[i1, k1, t] == 0 or model.x_param[i2, k2, t] == 0):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        if(i1 == i2 or k1 == k2 or model.a[i1] * model.a[i2] == 0):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.gamma[i1] + model.p[i1] + sum(model.d[q, i1] * model.delta[q, i1, k1, t] for q in model.q) <= model.gamma[i2] + model.bigM[2] * (5 - model.beta[alpha, i1, t] - model.beta[alpha, i2, t] - model.x[i1, k1, t] - model.x[i2, k2, t] - model.Lambda[i1, i2, t])

    def end_of_day_rule(self, model, i, k, t):
        if(model.x_param[i, k, t] == 0):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        if(model.tau[model.specialty[i], k, t] == 0):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.gamma[i] + model.p[i] + sum(model.d[q, i] * model.delta[q, i, k, t] for q in model.q) <= model.s[k, t]

    def time_ordering_precedence_rule(self, model, i1, i2, k, t):
        if(model.x_param[i1, k, t] == 0 or model.x_param[i2, k, t] == 0):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        if(i1 == i2
           or (model.specialty[i1] != model.specialty[i2])
           or (model.tau[model.specialty[i1], k, t] == 0)):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.gamma[i1] + model.p[i1] + sum(model.d[q, i1] * model.delta[q, i1, k, t] for q in model.q) <= model.gamma[i2] + model.bigM[2] * (3 - model.x[i1, k, t] - model.x[i2, k, t] - model.y[i1, i2, k, t])

    def start_time_ordering_priority_rule(self, model, i1, i2, k, t):
        if(model.x_param[i1, k, t] == 0 or model.x_param[i2, k, t] == 0):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        if(i1 == i2 or model.u[i1, i2] == 0
           or (model.specialty[i1] != model.specialty[i2])
           or (model.tau[model.specialty[i1], k, t] == 0)):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.gamma[i1] * model.u[i1, i2] <= model.gamma[i2] * (1 - model.u[i2, i1]) + model.bigM[2] * (2 - model.x[i1, k, t] - model.x[i2, k, t])

    def exclusive_precedence_rule(self, model, i1, i2, k, t):
        if(model.specialty[i1] != model.specialty[i2]):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        if(model.x_param[i1, k, t] == 0 or model.x_param[i2, k, t] == 0):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        if(i1 >= i2
           or (model.specialty[i1] != model.specialty[i2])
           or (model.tau[model.specialty[i1], k, t] == 0)):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.y[i1, i2, k, t] + model.y[i2, i1, k, t] == 1

    def lambda_rule(self, model, i1, i2, t):
        if(i1 >= i2 or not (model.a[i1] == 1 and model.a[i2] == 1)):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        # if patients not on same day
        if(sum(model.x_param[i1, k, t] for k in model.k) == 0 or sum(model.x_param[i2, k, t] for k in model.k) == 0):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.Lambda[i1, i2, t] + model.Lambda[i2, i1, t] == 1

    def extend_data(self, data):
        x_param_dict = {}
        for i in self.MP_instance.i:
            for t in self.MP_instance.t:
                # if patient is planned for day t, we allow her to be free in that day
                if(sum(round(self.MP_instance.x[i, k, t].value) for k in self.MP_instance.k) == 1):
                    for k in range(1, self.MP_instance.K + 1):
                        x_param_dict[(i, k, t)] = -1 # FREE
                # otherwise she is discarded, i.e. we do not allow her to be re-planned to another day t' != t
                else:
                    for k in range(1, self.MP_instance.K + 1):
                        x_param_dict[(i, k, t)] = 0 # DISCARDED
        data[None]['x_param'] = x_param_dict

    def fix_SP_variables(self):
        print("Fixing x variables for phase two...")
        fixed = 0
        for k in self.MP_instance.k:
            for t in self.MP_instance.t:
                for i in self.MP_instance.i:
                    if(self.SP_instance.x_param[i, k, t] == 0):
                        self.SP_instance.x[i, k, t].fix(0)
                        for q in self.SP_instance.q:
                            self.SP_instance.delta[q, i, k, t].fix(0)
                        fixed += 1
        print(str(fixed) + " x variables fixed.")

    def fix_MP_delta_variables(self):
        print("Fixing delta variables for phase one...")
        fixed = 0
        for q in self.MP_instance.q:
            for k in self.MP_instance.k:
                for t in self.MP_instance.t:
                    for i in self.MP_instance.i:
                        if(self.MP_instance.tau[self.MP_instance.specialty[i], k, t] == 0):
                            self.MP_instance.delta[q, i, k, t].fix(0)
                            fixed += 1
        print(str(fixed) + " delta variables fixed.")

    def fix_MP_x_variables(self):
        print("Fixing x variables for phase one...")
        fixed = 0
        for k in self.MP_instance.k:
            for t in self.MP_instance.t:
                for i in self.MP_instance.i:
                    if(self.MP_instance.tau[self.MP_instance.specialty[i], k, t] == 0):
                        self.MP_instance.x[i, k, t].fix(0)
                        fixed += 1
        print(str(fixed) + " x variables fixed.")


class VanillaLBBDPlanner(LBBDPlanner):

    def anesthetist_no_overlap_rule(self, model, i1, i2, k1, k2, t, alpha):
        if(model.x_param[i1, k1, t] + model.x_param[i2, k2, t] < 2):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        if(i1 == i2 or k1 == k2 or model.a[i1] * model.a[i2] == 0):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.gamma[i1] + model.p[i1] + sum(model.d[q, i1] * model.delta[q, i1, k1, t] for q in model.q) <= model.gamma[i2] + model.bigM[2] * (5 - model.beta[alpha, i1, t] - model.beta[alpha, i2, t] - model.x[i1, k1, t] - model.x[i2, k2, t] - model.Lambda[i1, i2, t])

    def end_of_day_rule(self, model, i, k, t):
        if(model.x_param[i, k, t] == 0):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        if(model.tau[model.specialty[i], k, t] == 0):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.gamma[i] + model.p[i] + sum(model.d[q, i] * model.delta[q, i, k, t] for q in model.q) <= model.s[k, t]

    def time_ordering_precedence_rule(self, model, i1, i2, k, t):
        if( model.x_param[i1, k, t] + model.x_param[i2, k, t] < 2):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        if(i1 == i2
           or (model.specialty[i1] != model.specialty[i2])
           or (model.tau[model.specialty[i1], k, t] == 0)):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.gamma[i1] + model.p[i1] + sum(model.d[q, i1] * model.delta[q, i1, k, t] for q in model.q) <= model.gamma[i2] + model.bigM[2] * (3 - model.x[i1, k, t] - model.x[i2, k, t] - model.y[i1, i2, k, t])

    def start_time_ordering_priority_rule(self, model, i1, i2, k, t):
        if( model.x_param[i1, k, t] + model.x_param[i2, k, t] < 2):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        if(i1 == i2 or model.u[i1, i2] == 0
           or (model.specialty[i1] != model.specialty[i2])
           or (model.tau[model.specialty[i1], k, t] == 0)):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.gamma[i1] * model.u[i1, i2] <= model.gamma[i2] * (1 - model.u[i2, i1]) + model.bigM[2] * (2 - model.x[i1, k, t] - model.x[i2, k, t])

    def exclusive_precedence_rule(self, model, i1, i2, k, t):
        if(model.x_param[i1, k, t] + model.x_param[i2, k, t] < 2):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        if(i1 >= i2
           or (model.specialty[i1] != model.specialty[i2])
           or (model.tau[model.specialty[i1], k, t] == 0)):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.y[i1, i2, k, t] + model.y[i2, i1, k, t] == 1

    def lambda_rule(self, model, i1, i2, t):
        if(i1 >= i2 or not (model.a[i1] == 1 and model.a[i2] == 1)):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        # if patients not on same day
        if(sum(model.x_param[i1, k, t] for k in model.k) == 0 or sum(model.x_param[i2, k, t] for k in model.k) == 0):
            self.discarded_constraints += 1
            return pyo.Constraint.Skip
        self.generated_constraints += 1
        return model.Lambda[i1, i2, t] + model.Lambda[i2, i1, t] == 1

    def extend_data(self, data):
        x_param_dict = {}
        for i in range(1, self.MP_instance.I + 1):
            for k in range(1, self.MP_instance.K + 1):
                for t in range(1, self.MP_instance.T + 1):
                    if(round(self.MP_instance.x[i, k, t].value) == 1):
                        x_param_dict[(i, k, t)] = 1
                    else:
                        x_param_dict[(i, k, t)] = 0
        data[None]['x_param'] = x_param_dict

    def fix_SP_variables(self):
        print("Fixing x variables for phase two...")
        fixed = 0
        for k in self.MP_instance.k:
            for t in self.MP_instance.t:
                for i in self.MP_instance.i:
                    self.SP_instance.x[i, k, t].fix(round(self.MP_instance.x[i, k, t].value))
                    for q in self.SP_instance.q:
                        self.SP_instance.delta[q, i, k, t].fix(round(self.MP_instance.delta[q, i, k, t].value))
                    fixed += 1
        print(str(fixed) + " x variables fixed.")


class Solution:

    def __init__(self, model_instance=None):
        if model_instance:
            self.extract_solution(model_instance)

    def extract_solution(self, model_instance):
        self.I = model_instance.I
        self.J = model_instance.J
        self.K = model_instance.K
        self.T = model_instance.T
        self.A = model_instance.A
        self.Q = model_instance.Q

        # x, beta and delta: discard variables set to 0
        self.x = {key: value for key, value in model_instance.x.extract_values().items() if round(value) != 0}
        self.beta = {key: value for key, value in model_instance.beta.extract_values().items() if round(value) != 0}
        self.gamma = model_instance.gamma.extract_values()
        self.delta = {key: value for key, value in model_instance.delta.extract_values().items() if round(value) != 0}

        # parameters
        self.d = model_instance.d.extract_values()
        self.c = model_instance.c.extract_values()
        self.a = model_instance.a.extract_values()
        self.specialty = model_instance.specialty.extract_values()
        self.r = model_instance.r.extract_values()
        self.s = model_instance.s.extract_values()
        self.p = model_instance.p.extract_values()
        self.tau = model_instance.tau.extract_values()
        self.precedence = model_instance.precedence.extract_values()

        self.objective_value = pyo.value(model_instance.objective)

    def to_patients_dict(self):
        patients_dict = {(k, t): [] for k in range(1, self.K + 1) for t in range(1, self.T + 1)}
        for (i, k, t) in self.x:
            delay = False
            arrival_delay = 0
            for q in range(1, self.Q + 1):
                if (q, i, k, t) in self.delta:
                    delay = True
                    arrival_delay = self.d[(q, i)]
                    break
            anesthetist = 0
            for alpha in range(1, self.A + 1):
                if (alpha, i, t) in self.beta:
                    anesthetist = alpha
                    break
            patients_dict[(k, t)].append(Patient(id=i, 
                                                 priority=self.r[i],
                                                 room=k,
                                                 specialty=self.specialty[i],
                                                 day=t,
                                                 operatingTime=self.p[i],
                                                 arrival_delay=arrival_delay,
                                                 covid=self.c[i], 
                                                 precedence=self.precedence[i], 
                                                 delayWeight=None, 
                                                 anesthesia=self.a[i], 
                                                 anesthetist=anesthetist, 
                                                 order=round(self.gamma[i], 2),
                                                 delay=delay)
                                        )
        return patients_dict
