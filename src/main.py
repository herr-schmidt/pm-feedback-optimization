from data_loader import DataLoader
from planners import HeuristicLBBDPlanner
from utils import SolutionVisualizer

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