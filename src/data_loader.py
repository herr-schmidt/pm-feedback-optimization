import pandas as pd
from math import floor, ceil
import numpy as np
import tables

class DataLoader:

    def __init__(self):
        self.input_data_frame = None

        self.operating_times = {}
        self.delay_times = {}
        self.priority_scores = {}
        self.anesthesiae = {}
        self.infections = {}
        self.relative_precedences = {}

        self.patients_ids = {}
        self.patients_specialties = {}
        self.procedures_types = {} # clean/dirty

        np.random.seed(seed=781015)

    def generate_solver_input(self):
        # for now we assume same duration for each room, on each day
        max_OR_time = tables.opening_time
        return {
            None: {
                'I': {None: len(self.patients_ids)},
                'J': {None: 1},
                'K': {None: 1},
                'T': {None: 5},
                'A': {None: 0},
                'M': {None: 7},
                'Q': {None: 1},
                's': tables.operating_slots_table,
                'An': None,
                'Gamma': tables.robustness_table,
                'tau': tables.tau_table,
                'p': self.operating_times,
                'd': self.delay_times,
                'r': self.priority_scores,
                'a': self.anesthesiae,
                'c': self.infections,
                'u': self.relative_precedences,
                'patientId': self.patients_ids,
                'specialty': self.patients_specialties,
                'precedence': self.procedures_types,
                'bigM': {
                    1: floor(max_OR_time/min(self.operating_times)),
                    2: max_OR_time
                }
            }
        }

    def load_input_file(self, filepath):
        self.input_data_frame = pd.read_excel(io=filepath, sheet_name="patients")
        print(self.input_data_frame)

        solver_patients_ids = [solver_patient_id for solver_patient_id in range(1, len(self.input_data_frame) + 1)]

        self.operating_times = {solver_patient_id: operating_time for (solver_patient_id, operating_time) in zip(solver_patients_ids, self.input_data_frame["Prediction"])}
        self.delay_times = {solver_patient_id: (staff_estimate - self.operating_times[solver_patient_id] if staff_estimate - self.operating_times[solver_patient_id] > 0 else 0) for (solver_patient_id, staff_estimate) in zip(solver_patients_ids, self.input_data_frame["Staff_estimate"])}
        self.priority_scores = {solver_patient_id: priority_score for (solver_patient_id, priority_score) in zip(solver_patients_ids, np.random.uniform(low=10, high=120, size=len(solver_patients_ids)))}
        self.anesthesiae = {solver_patient_id: 0 for solver_patient_id in solver_patients_ids}
        self.infections = {solver_patient_id: 0 for solver_patient_id in solver_patients_ids}

        self.patients_ids = {solver_patient_id: actual_patient_id for (solver_patient_id, actual_patient_id) in zip(solver_patients_ids, self.input_data_frame["Id"])}
        self.patients_specialties = {}
        self.relative_precedences = {}
        self.procedures_types = {} # clean/dirty
        
    def compute_patients_specialties(self, procedures):
        pass