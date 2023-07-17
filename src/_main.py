from feedback_optimizer import MLFeedbackOptimizer
from pprint import pprint

ML_feedback_optimizer = MLFeedbackOptimizer()

# Phase 1
ML_feedback_optimizer.create_model(predict_operating_times=True)
ML_feedback_optimizer.solve_model()
ML_feedback_optimizer.visualize_solution()
utilization_and_overtime = ML_feedback_optimizer.compute_real_world_utilization_and_overtime()
pprint(utilization_and_overtime)

ML_feedback_optimizer.extract_selected_patients_dict()
pprint(ML_feedback_optimizer.selected_patients)

print(ML_feedback_optimizer.selected_patients)
ML_feedback_optimizer.check_overtime()

selected_patients_ids = ML_feedback_optimizer.extract_patient_ids()
pprint(selected_patients_ids)

pprint(ML_feedback_optimizer.fixed_ORs)

# first trial: reoptimize with cut, up to N times
N = 100
iteration = 0
# only go on if within iterations limit or some slot is still not fixed
while iteration < N and sum(ML_feedback_optimizer.fixed_ORs.values()) < 15:
    ML_feedback_optimizer.fix_model_ORs()
    for (k, t) in ML_feedback_optimizer.fixed_ORs.keys():
        if ML_feedback_optimizer.fixed_ORs[(k, t)] == 0:
            ML_feedback_optimizer.planner.MP_instance.patients_cuts.add(sum(1 - ML_feedback_optimizer.planner.MP_instance.x[i, k, t] for i in ML_feedback_optimizer.planner.MP_instance.i if round(ML_feedback_optimizer.planner.MP_instance.x[i, k, t].value) == 1) >= 1)
    
    # re-solve
    ML_feedback_optimizer.solve_model()
    # ML_feedback_optimizer.visualize_solution()
    utilization_and_overtime = ML_feedback_optimizer.compute_real_world_utilization_and_overtime()
    pprint(utilization_and_overtime)

    ML_feedback_optimizer.extract_selected_patients_dict()
    pprint(ML_feedback_optimizer.selected_patients)

    print(ML_feedback_optimizer.selected_patients)
    ML_feedback_optimizer.check_overtime()

    selected_patients_ids = ML_feedback_optimizer.extract_patient_ids()
    pprint(selected_patients_ids)

    pprint(ML_feedback_optimizer.fixed_ORs)
    
    iteration += 1

print(iteration)
# pprint(ML_feedback_optimizer.date_t_map)
 