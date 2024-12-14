import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt

COMBINED_DF_PATH = 'combined.parquet'


def main():
    full_df = pandas.read_parquet(COMBINED_DF_PATH)

    occupant_df = get_occupant_df(full_df)

    occupant_df = get_injury_severities(occupant_df)

    occupant_df = convert_columns_to_numeric(occupant_df)

    plot_correlations(occupant_df)


def get_occupant_df(full_df):
    occupants_list = []

    def extract_occupant_data(row):
        occupants = row['Occupants']

        for occupant in occupants:
            occupant['CaseId'] = row['CaseId']
            occupant['CrashTime'] = row['CrashTime']
            occupant['InTransportVehCount'] = row['InTransportVehCount']
            occupant['Fatal'] = row['Fatal']

            occupants_list.append(occupant)

    full_df.apply(extract_occupant_data, axis=1)

    return pandas.DataFrame(occupants_list)


def get_injury_severities(occupant_df):
    def get_individual_severity(row):
        max_sev_code = row['MaxSevCode']
        injured_status_desc = row['InjuredStatusDesc']

        try:
            if 'Not Injured' in injured_status_desc:
                return 0

            if 'Unknown' in injured_status_desc:
                return numpy.nan

            if int(max_sev_code) > 6:
                return numpy.nan

            return int(max_sev_code)

        except:
            return numpy.nan

    occupant_df['InjurySeverity'] = occupant_df.apply(get_individual_severity, axis=1)

    return occupant_df


def convert_columns_to_numeric(occupant_df):
    columns_to_drop = [
        'MaxSevCode',
        'ISS',
        'InjuredStatus',
        'Treatment',
        'FacilityType',
        'IsSCIVehicle',
        'OCCID',
        'VEHID',
        'CaseId',
        'CASEID',
        'Loc',
        'OCCNUM',
        'VEHNUM',
    ]

    occupant_df = occupant_df.drop(columns_to_drop, axis=1)

    almost_numeric_columns = {
        'AgeInYears': ' Years',
        'Height': ' cm',
        'Weight': ' kgs',
    }

    def convert_to_numeric(value):
        try:
            return float(value)
        except:
            return numpy.nan

    for column in occupant_df.columns:

        if occupant_df[column].dtype in ['float64', 'int64']:
            continue

        if column not in almost_numeric_columns.keys():
            continue

        suffix = almost_numeric_columns.get(column, '')

        if suffix:
            occupant_df[column] = occupant_df[column].str.replace(suffix, '').apply(convert_to_numeric)

    def convert_crash_time_to_numeric(crash_time):
        try:
            hours = int(crash_time.split(':')[0])
            minutes = int(crash_time.split(':')[1])
            return hours * 60 + minutes
        except:
            return numpy.nan

    occupant_df['CrashTime'] = occupant_df['CrashTime'].apply(convert_crash_time_to_numeric)

    return occupant_df


def plot_correlations(occupant_df):
    correlation_matrix = occupant_df.corr(numeric_only=True)

    plt.figure(figsize=(30, 30))
    seaborn.heatmap(correlation_matrix, annot=True)
    plt.title('Full Occupant-Related Feature Correlation Matrix')
    plt.show()

    target_correlations = correlation_matrix['InjurySeverity'].drop('InjurySeverity').dropna().sort_values(
        ascending=True)

    plt.figure(figsize=(10, 8))
    target_correlations.plot(kind='barh',
                             color=target_correlations.apply(lambda x: 'tab:orange' if x < 0 else 'tab:blue'))
    plt.title('Top Feature Correlations with Occupant Injury Severity')
    plt.xlabel('Correlation Value')
    plt.ylabel('Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
