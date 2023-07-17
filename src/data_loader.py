import pandas as pd
from math import floor, ceil
import numpy as np
import tables
from enum import Enum

class SurgeryType(Enum):
    CLEAN = 1
    DIRTY = 2
    COVID = 3

class DataLoader:

    def __init__(self):
        self.input_data_frame = None
        self.data_consistency = False

        self.solver_patients_ids = []

        self.operating_times = {}
        self.delay_times = {}
        self.priority_scores = {}
        self.anesthesiae = {}
        self.infections = {}
        self.relative_precedences = {}

        self.patients_ids = {}
        self.patients_procedures = {}
        self.patients_specialties = {}
        self.procedures_types = {} # clean/dirty
        self.actual_operating_times = {}

        np.random.seed(seed=781015)

    def generate_solver_input(self):
        if not self.data_consistency:
            raise Exception("Data is not consistent. Please load an input file first.")
        
        # for now we assume same duration for each room, on each day
        max_OR_time = tables.opening_time
        return {
            None: {
                'I': {None: len(self.patients_ids)},
                'J': {None: 1},
                'K': {None: 1},
                'T': {None: 15},
                'A': {None: 1},
                'M': {None: 7},
                'Q': {None: 1},
                's': tables.operating_slots_table,
                'An': {(1, 1): max_OR_time, (1, 2): max_OR_time, (1, 3): max_OR_time, (1, 4): max_OR_time, (1, 5): max_OR_time},
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

    def load_input_file(self, filepath, predicted_operating_times=True):
        self.input_data_frame = pd.read_excel(io=filepath, sheet_name="patients")
        print(self.input_data_frame)

        self.solver_patients_ids = [solver_patient_id for solver_patient_id in range(1, len(self.input_data_frame) + 1)]

        if predicted_operating_times: # consider as operating times the predictions provided by PM1
            self.operating_times = {solver_patient_id: operating_time for (solver_patient_id, operating_time) in zip(self.solver_patients_ids, self.input_data_frame["Prediction"])}
        else: # consider operating times provided by medical personnel
            self.operating_times = {solver_patient_id: operating_time for (solver_patient_id, operating_time) in zip(self.solver_patients_ids, self.input_data_frame["Staff_estimate"])}
        
        self.delay_times = {(1, solver_patient_id): 30 for solver_patient_id in self.solver_patients_ids}

        self.priority_scores = {solver_patient_id: priority_score for (solver_patient_id, priority_score) in zip(self.solver_patients_ids, np.random.uniform(low=10, high=120, size=len(self.solver_patients_ids)))}
        self.anesthesiae = {solver_patient_id: 0 for solver_patient_id in self.solver_patients_ids}
        self.infections = {solver_patient_id: 0 for solver_patient_id in self.solver_patients_ids}

        self.patients_ids = {solver_patient_id: actual_patient_id for (solver_patient_id, actual_patient_id) in zip(self.solver_patients_ids, self.input_data_frame["Id"])}
        self.patients_procedures = {solver_patient_id: procedures.split("|") for (solver_patient_id, procedures) in zip(self.solver_patients_ids, self.input_data_frame["Procedures"])}
        
        self.patients_specialties = {solver_patient_id: 1 for solver_patient_id in self.solver_patients_ids}
        self.compute_patients_procedures_types() # clean/dirty
        self.compute_relative_precedences()

        self.actual_operating_times = {solver_patient_id: actual_operating_time for (solver_patient_id, actual_operating_time) in zip(self.solver_patients_ids, self.input_data_frame["Actual_time"])}

        self.data_consistency = True
        
    def compute_patients_procedures_types(self):
        for patient in self.patients_procedures.keys():
            self.procedures_types[patient] = 1 # assume clean
            for procedure in self.patients_procedures[patient]:
                if tables.dirty_mapping_table[procedure] == 1:
                    self.procedures_types[patient] = 3
                    break

    def compute_relative_precedences(self):
        for i1 in self.solver_patients_ids:
            for i2 in self.solver_patients_ids:
                if self.procedures_types[i1] < self.procedures_types[i2]:
                    self.relative_precedences[(i1, i2)] = 1
                else:
                    self.relative_precedences[(i1, i2)] = 0