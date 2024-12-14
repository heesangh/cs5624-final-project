import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn

COMBINED_DF_PATH = 'combined.parquet'


def main():
    full_df = pandas.read_parquet(COMBINED_DF_PATH)

    vehicle_df = get_vehicle_df(full_df)

    vehicle_df = convert_columns_to_numeric(vehicle_df)

    plot_correlations(vehicle_df)


def get_vehicle_df(full_df):
    vehicles_list = []

    def extract_vehicle_data(row):
        vehicles = row['Vehicles']

        for vehicle in vehicles:
            vehicle['CaseId'] = row['CaseId']
            vehicle['CrashTime'] = row['CrashTime']
            vehicle['InTransportVehCount'] = row['InTransportVehCount']
            vehicle['Fatal'] = row['Fatal']
            vehicle['Summary'] = row['Summary']

            vehicles_list.append(vehicle)

    full_df.apply(extract_vehicle_data, axis=1)

    return pandas.DataFrame(vehicles_list)


def convert_columns_to_numeric(vehicle_df):
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
        'LengthDesc': ' cm',
        'MaxDVXDesc': ' kmph',
        'MaxDVYDesc': ' kmph',
        'MmntArmDesc': ' cm',
        'ModelYear': '',
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
        'InTransportVehCount': '',
    }

    columns_to_drop = [column for column in vehicle_df.columns if column.endswith('Form')]
    vehicle_df = vehicle_df.drop(columns_to_drop, axis=1)

    vehicle_severity_mapping = {
        'Unknown': 0,
        'Light ': 1,
        'Moderate ': 2,
        'Severe': 3
    }

    vehicle_df['CrashSeverityDesc'] = vehicle_df['CrashSeverityDesc'].map(vehicle_severity_mapping)

    def convert_to_numeric(value):
        try:
            return float(value)
        except:
            return numpy.nan

    for column in vehicle_df.columns:

        if vehicle_df[column].dtype in ['float64', 'int64']:
            continue

        if column not in almost_numeric_columns.keys():
            continue

        suffix = almost_numeric_columns.get(column, '')

        if suffix:
            vehicle_df[column] = vehicle_df[column].str.replace(suffix, '').apply(convert_to_numeric)

    return vehicle_df


def plot_correlations(vehicle_df):
    correlation_matrix = vehicle_df.corr(numeric_only=True)

    plt.figure(figsize=(30, 30))
    seaborn.heatmap(correlation_matrix, annot=True)
    plt.title('Full Vehicle-Related Feature Correlation Matrix')
    plt.show()

    target_correlations = correlation_matrix['CrashSeverityDesc'].drop('CrashSeverityDesc').dropna().sort_values(
        ascending=True)

    target_correlations = pandas.concat([target_correlations[:5], target_correlations[-5:]])

    plt.figure(figsize=(10, 8))
    target_correlations.plot(kind='barh',
                             color=target_correlations.apply(lambda x: 'tab:orange' if x < 0 else 'tab:blue'))
    plt.title('Top Feature Correlations with Vehicle Damage Severity')
    plt.xlabel('Correlation Value')
    plt.ylabel('Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
