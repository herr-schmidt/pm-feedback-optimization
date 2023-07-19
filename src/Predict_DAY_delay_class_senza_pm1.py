import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import datetime
import pickle
from collections import Counter
from sklearn import metrics


class day_delay_predictor:
    def __init__(self):
        self.schedule_dict_example = {datetime.datetime(2022, 9, 5, 0, 0): [331, 353, 361, 366, 375, 386],
                                      datetime.datetime(2022, 9, 6, 0, 0): [328, 329, 352, 374, 380, 385],
                                      datetime.datetime(2022, 9, 7, 0, 0): [335, 343, 348, 373, 398],
                                      datetime.datetime(2022, 9, 8, 0, 0): [326, 340, 377, 389, 396],
                                      datetime.datetime(2022, 9, 9, 0, 0): [327, 359, 381, 382, 399],
                                      datetime.datetime(2022, 9, 12, 0, 0): [334, 336, 347, 371, 379],
                                      datetime.datetime(2022, 9, 13, 0, 0): [342, 345, 350, 378, 397],
                                      datetime.datetime(2022, 9, 14, 0, 0): [344, 346, 367, 372, 383],
                                      datetime.datetime(2022, 9, 15, 0, 0): [338, 368, 370, 376, 388],
                                      datetime.datetime(2022, 9, 16, 0, 0): [325, 356, 360, 364, 395],
                                      datetime.datetime(2022, 9, 19, 0, 0): [337, 355, 392, 393, 394],
                                      datetime.datetime(2022, 9, 20, 0, 0): [324, 332, 363, 387],
                                      datetime.datetime(2022, 9, 21, 0, 0): [341, 354, 357, 362],
                                      datetime.datetime(2022, 9, 22, 0, 0): [330, 333, 351, 391],
                                      datetime.datetime(2022, 9, 23, 0, 0): [339, 365, 390]}
        self.MAX_NUMBER_OF_PATIENTS = 10
        # load patients information from disk
        patients_file = os.path.join("test_set_encoded_patients.csv")
        self.df_patients = pd.read_csv(patients_file, sep=';', decimal=",", encoding='latin-1', skipinitialspace=True, parse_dates=['Data'])
        # load the model from disk
        model_file = os.path.join('DAY_delay_model2023_07_18-10_29_02_AM.sav')
        self.model = pickle.load(open(model_file, 'rb'))

    def reschedule_patients(self, schedule_dict):
        # set index
        df_patients = self.df_patients.set_index('Unnamed: 0')
        df_patients_rescheduled = pd.DataFrame()

        for day, patient_list in schedule_dict.items():
            day_list = df_patients.loc[patient_list].copy()
            day_list['Data'] = day
            day_list['Weekday'] = day.weekday()
            df_patients_rescheduled = pd.concat([df_patients_rescheduled, day_list])

        return df_patients_rescheduled

    def encode_days(self, df_patients):
        # it basically provides the complex encoding for each day
        # unique trace attributes are taken as is, 'sum' one are summed over all rows

        # tempi previsti di esecuzione sommati come attributi di traccia
        trace_attributes = {"episodio_data": "unique", "Weekday": "unique", "Data": "unique",
                            "ritardo_sala_e_ripristino": "sum",
                            "Tempo_medico_previsto": "sum", "Tempo_impiego_ripristino_sala_previsto": "sum",
                            # "Tempo_impiego_ripristino_sala_pm1": "sum",
                            "Tempo_impiego_ripristino_sala_effettivo": "sum", "overtime": "sum",
                            "overtime_minus_pm1": "sum", "overtime_wrt_tempi_medici": "sum"}
        event_attributes = {"Sesso": "nominal", "TACE": "int", "TARE": "int",
                            "UO_supersimple": "nominal", "Regime_di_ricovero": "nominal",
                            "Tecnica_Utilizzata": "nominal",
                            "Anno_di_nascita": "int"}
        events_columns = event_attributes.keys()

        trace_attributes_number = len(trace_attributes.keys())
        encoding_length = trace_attributes_number + self.MAX_NUMBER_OF_PATIENTS * (len(df_patients.columns) - trace_attributes_number)

        df_grouped = df_patients.groupby(by=["Data"], dropna=False)
        groups_stats = dict(sorted(Counter(df_grouped['episodio_data'].count().to_list()).items()))

        df_day_encoded = pd.DataFrame()
        # df_day_encoded2 = pd.DataFrame()
        for key, df in df_grouped:
            df = df.reset_index(drop=True)
            # compute trace attributes
            newrow = pd.Series()
            for column, type in trace_attributes.items():
                if type == 'unique':
                    value = df[column].to_list()[0]
                elif type == 'sum':
                    value = df[column].sum()
                newrow = pd.concat([newrow, pd.Series({column: value})])

            # compute event attributes
            df = df[events_columns]
            for index, row in df.iterrows():
                rename_columns_dict = {v: v + '_' + str(index) for v in events_columns}
                row_to_add = row.rename(rename_columns_dict)
                newrow = pd.concat([newrow, row_to_add])
            row.loc[:] = 0  # set the padding, be careful using 0 may not be a good padding!
            for i in range(index + 1, self.MAX_NUMBER_OF_PATIENTS):
                rename_columns_dict = {v: v + '_' + str(i) for v in events_columns}
                row_to_add = row.rename(rename_columns_dict)
                newrow = pd.concat([newrow, row_to_add])
            df_day_encoded = df_day_encoded.append(newrow, ignore_index=True)
            # df_day_encoded2 = pd.concat([df_day_encoded2, newrow.to_frame().T], ignore_index=True)  # change object to int!

        return df_day_encoded

    def define_label(self, df):
        options = ['ritardo_sala_e_ripristino', 'Tempo_impiego_ripristino_sala_effettivo', 'overtime',
                   'overtime_minus_pm1', 'overtime_wrt_tempi_medici']
        df['label'] = df['overtime_wrt_tempi_medici']
        df = df.drop(columns=options)
        return df

    def get_day_prediction(self, schedule_dict):
        # get reschedule df
        df_patients_rescheduled = self.reschedule_patients(schedule_dict)
        # encode df per day
        df_day_rescheduled = self.encode_days(df_patients_rescheduled)
        # define label
        df_day_rescheduled = self.define_label(df_day_rescheduled)
        # define test sets
        Days_test = df_day_rescheduled['Data']
        X_test = df_day_rescheduled.drop(columns=['episodio_data', 'Data', 'label'])
        Y_test = df_day_rescheduled['label']
        # do inference
        Y_pred = self.model.predict(X_test)
        # compute and print MAE for reference
        # print('MAE:', metrics.mean_absolute_error(Y_test, Y_pred))
        # construct outcome dictionary
        Y_day_predictions = pd.concat([Days_test, pd.Series(data=Y_pred, name='pred')], axis=1)
        Day_prediction_dict = Y_day_predictions.set_index('Data').to_dict()['pred']

        return Day_prediction_dict


