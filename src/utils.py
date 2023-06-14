import plotly.express as px
import pandas as pd
import datetime


class SolutionVisualizer:

    def __init__(self):
        pass

    def compute_solution_value(self, solution):
        KT = max(solution.keys())
        K = KT[0]
        T = KT[1]

        value = 0
        for t in range(1, T + 1):
            for k in range(1, K + 1):
                for patient in solution[(k, t)]:
                    value = value + patient.priority
        return value

    def compute_solution_partitioning_by_precedence(self, solution):
        KT = max(solution.keys())
        K = KT[0]
        T = KT[1]

        PO = 0
        PR = 0
        SO = 0
        SR = 0
        CO = 0
        CR = 0
        for t in range(1, T + 1):
            for k in range(1, K + 1):
                for patient in solution[(k, t)]:
                    if(patient.precedence == 1):
                        PO += 1
                    elif(patient.precedence == 2):
                        PR += 1
                    elif(patient.precedence == 3):
                        SO += 1
                    elif(patient.precedence == 4):
                        SR += 1
                    elif(patient.precedence == 5):
                        CO += 1
                    elif(patient.precedence == 6):
                        CR += 1

        return [PO, PR, SO, SR, CO, CR]


    def print_solution(self, solution):
        if(solution is None):
            print("No solution was found!")
            return

        KT = max(solution.keys())
        K = KT[0]
        T = KT[1]

        print("Operated patients, for each day and for each room:\n")

        operatedPatients = 0
        for t in range(1, T + 1):
            for k in range(1, K + 1):
                print("Day: " + str(t) + "; Operating Room: S" + str(k))
                if(len(solution[(k, t)]) == 0):
                    print("---")
                for patient in solution[(k, t)]:
                    print(patient)
                    operatedPatients += 1
                print("\n")
        print("Total number of operated patients: " + str(operatedPatients))

    def solution_as_string(self, solution):
        KT = max(solution.keys())
        K = KT[0]
        T = KT[1]

        result = "Operated patients, for each day and for each room:\n"
        for t in range(1, T + 1):
            for k in range(1, K + 1):
                result += "Day: " + str(t) + "; Operating Room: S" + str(k) + "\n"
                if(len(solution[(k, t)]) == 0):
                    result += "---" + "\n"
                for patient in solution[(k, t)]:
                    result += str(patient) + "\n"
                result += "\n"
        return result

    def count_operated_patients(self, solution):
        return len(solution.x)

    def plot_graph(self, solution):
        if(solution is None):
            print("No solution exists to be plotted!")
            return

        patients_as_data_frames = pd.DataFrame([])
        for (i, k, t) in solution.x.keys():
            start = datetime.datetime(2000, 1, t, 8, 0, 0) + datetime.timedelta(minutes=round(solution.gamma[i]))
            arrival_delay = solution.d[(1, i)] if (1, i, k, t) in solution.delta else 0
            finish = start + datetime.timedelta(minutes=round(solution.p[i])) + datetime.timedelta(minutes=round(arrival_delay))
            room = "S" + str(k)
            covid = "Y" if solution.c[i] == 1 else "N"
            precedence = solution.precedence[i]
            anesthesia = "Y" if solution.a[i] == 1 else "N"
            anesthetist = ""
            for alpha in range(1, solution.A + 1):
                if (alpha, i, t) in solution.beta:
                    anesthetist = "A" + str(alpha)
            delay = True if (1, i, k, t) in solution.delta else False
            if(precedence == 1):
                precedence = "Clean procedure"
            elif(precedence == 3):
                precedence = "Dirty procedure"
            elif(precedence == 5):
                precedence = "Covid-19 patient"
            
            patient_data_frame = pd.DataFrame([dict(Start=start, Finish=finish, Room=room, Covid=covid, Precedence=precedence, Anesthesia=anesthesia, Anesthetist=anesthetist, Delay=delay)])
            patients_as_data_frames = pd.concat([patients_as_data_frames, patient_data_frame])
            

        # sort legend's labels
        sortingOrder = ["Clean procedure",
                        "Dirty procedure",
                        "Covid-19 patient"]
        order = []
        for precedenceValue in patients_as_data_frames["Precedence"].tolist():
            if(not precedenceValue in order):
                order.append(precedenceValue)

        order.sort(key=sortingOrder.index)
        patients_as_data_frames = patients_as_data_frames.set_index('Precedence')
        patients_as_data_frames= patients_as_data_frames.T[order].T.reset_index()

        color_discrete_map = {'Clean procedure': '#38A6A5', 
                                'Dirty procedure': '#73AF48',
                                'Covid-19 patient': '#E17C05'}

        fig = px.timeline(patients_as_data_frames,
                          x_start="Start",
                          x_end="Finish",
                          y="Room",
                          color="Precedence",
                          text="Anesthetist",
                          labels={"Start": "Procedure start", "Finish": "Procedure end", "Room": "Operating room",
                                  "Covid": "Covid patient", "Precedence": "Procedure Type and Delay", "Anesthesia": "Need for anesthesia", "Anesthetist": "Assigned anesthetist"},
                          hover_data=["Anesthesia", "Anesthetist", "Precedence", "Covid", "Delay"],
                          color_discrete_map=color_discrete_map
                          )

        rangebreaks = []
        current_day = datetime.datetime(2000, 1, 1, 8, 0, 0)

        operating_day_span = solution.s[(1, 1)] # assume every (k, t) slot has the same duration

        for _ in range(1, solution.T + 0):
            rb = dict(bounds=[str(current_day + datetime.timedelta(minutes=operating_day_span)), str(current_day + datetime.timedelta(days=1))])
            rangebreaks.append(rb)

            # day separator
            fig.add_vline(x=str(current_day), line_width=1, line_dash="solid", line_color="black")

            current_day = current_day + datetime.timedelta(days=1)

        fig.update_xaxes(rangebreaks=rangebreaks)

        # last two separator lines
        fig.add_vline(x=str(current_day + datetime.timedelta(minutes=operating_day_span) - datetime.timedelta(days=1)), line_width=1, line_dash="solid", line_color="black")
        fig.add_vline(x=str(current_day + datetime.timedelta(minutes=operating_day_span)), line_width=1, line_dash="solid", line_color="black")

        fig.update_layout(xaxis=dict(title='Timetable', tickformat='%H:%M:%S',), legend={"traceorder": "normal"})
        fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
        ))
        fig.update_yaxes(categoryorder='category descending')
        fig.show()
