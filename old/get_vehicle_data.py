import pandas
import seaborn
import numpy
import matplotlib.pyplot as plt

# raw_df = pandas.read_parquet('combined.parquet')
#
# vehicles_list = []
#
#
# def extract_vehicle_data(row):
#     vehicles = row['Vehicles']
#
#     for vehicle in vehicles:
#         vehicle['CaseId'] = row['CaseId']
#         vehicle['CrashTime'] = row['CrashTime']
#         vehicle['InTransportVehCount'] = row['InTransportVehCount']
#         vehicle['Fatal'] = row['Fatal']
#         vehicle['Summary'] = row['Summary']
#
#         vehicles_list.append(vehicle)
#
#
# raw_df.apply(extract_vehicle_data, axis=1)

# df = pandas.DataFrame(vehicles_list)

df = pandas.read_excel('vehicles_filtered.xlsx')

almost_numeric_columns = {
    'AvgTrackDesc': ' cm',
    'BodyCat': '',
    'CargoWt': ' kgs',
    'CrashType': '',
    'CurbWt': ' kgs',
    'CylindersDesc': '',
    'DisplacementDesc': ' L',
    'EndWidthDesc': ' cm',
    'EstDistDesc': ' m',
    'FrontGAWR': ' kgs',
    'HeadAngleDesc': ' degrees',
    'HeadAngleOtherDesc': ' degrees',
    'HighBarrierDesc': ' kmph',
    'HighDVDesc': ' kmph',
    'HighDVLatDesc': ' kmph',
    'HighDVLongDesc': ' kmph',
    'HighEnergyDesc': ' joules',
    # 'InitialTravelLaneDesc': '',
    'LengthDesc': ' cm',
    'MaxDVXDesc': ' kmph',
    'MaxDVYDesc': ' kmph',
    'MmntArmDesc': ' cm',
    'ModelYear': '',
    # 'NumLanesDesc': '',
    'OCCNUM': '',
    'ObjectHeightDesc': '',
    'ObjectLengthDesc': '',
    'ObjectWidthDesc': '',
    'OverHangFrontDesc': ' cm',
    'OverHangRearDesc': ' cm',
    'QTurns': '',
    'RearGAWR': ' kgs',
    'ShoulderWidthDesc': '',
    'SpeedPosted': ' kmph',
    'TotGVWR': ' kgs',
    'VEHNUM': '',
    'WheelBaseDesc': ' cm',
    'WidthDesc': ' cm',
    'CaseId': '',
    # 'CrashTime': '',
    'InTransportVehCount': '',
}

for column in almost_numeric_columns:

    if df[column].dtype in ['int64', 'float64', 'bool']:
        continue

    raw = df[column]

    suffix = almost_numeric_columns.get(column, '')

    if suffix:
        raw = raw.str.replace(suffix, '')


    def convert_to_float(value):
        try:
            return float(value)
        except:
            return numpy.nan


    converted = raw.apply(convert_to_float)

    df[column] = converted

vehicle_severity_mapping = {
    'Unknown': 0,
    'Light ': 1,
    'Moderate ': 2,
    'Severe': 3
}

for column in df.columns:

    if column == 'CrashSeverityDesc':
        df[column] = df[column].map(vehicle_severity_mapping)
        continue

    # skip numeric/boolean columns
    if df[column].dtype in ['int64', 'float64', 'bool']:
        continue

    unique_values = list(df[column].unique().astype(str))

    unique_values = sorted(unique_values)

    value_mapping = {value: index for index, value in enumerate(unique_values)}
    df[column] = df[column].map(value_mapping)

correlation_matrix = df.corr(numeric_only=True)

vehicle_correlation = correlation_matrix['CrashSeverityDesc'].drop('CrashSeverityDesc').dropna().sort_values(
    ascending=True)

# keep only first 5 and last 5 using pandas.concat

vehicle_correlation = pandas.concat([vehicle_correlation[:5], vehicle_correlation[-5:]])

plt.figure(figsize=(10, 8))
vehicle_correlation.plot(kind='barh', color=vehicle_correlation.apply(lambda x: 'tab:orange' if x < 0 else 'tab:blue'))
plt.title('Feature Correlations with Vehicle Damage Severity')
plt.xlabel('Correlation Value')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(30, 30))
seaborn.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# correlation_matrix = df.corr()['CrashSeverityDesc'].drop('CrashSeverityDesc')
#
# # sort matrix
# correlation_matrix = correlation_matrix.sort_values(ascending=False)
#
# # print with labels
# for column in correlation_matrix.index:
#     # print with tab separator to space columns evenly
#     print(f'{column}\t{100 * correlation_matrix[column]:.2f}%')
#
# exit()
#
# # correlation_matrix = df.corr(numeric_only=True)
#
# # correlations = (
# #     correlation_matrix.where(numpy.tril(numpy.ones(correlation_matrix.shape), k=-1).astype(bool))
# #     # .stack()
# #     # .sort_values(ascending=False)
# # )
#
# plt.figure(figsize=(30, 30))
# seaborn.heatmap(correlation_matrix, annot=True)
# plt.title('Correlation Matrix')
# plt.show()

# plt.figure(figsize=(30, 30))
# seaborn.heatmap(correlations, annot=True)
# plt.title('Correlation Matrix')
# plt.show()

# correlations = (
#     correlation_matrix.where(numpy.tril(numpy.ones(correlation_matrix.shape), k=-1).astype(bool))
#     .stack()
#     .sort_values(ascending=False)
# )
#
# for correlation in correlations:
#     # if correlation value is between -0.1 and 0.1, skip
#     print(correlation)
