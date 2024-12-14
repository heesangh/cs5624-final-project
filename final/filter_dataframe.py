import numpy
import pandas

COMBINED_DF_PATH = 'combined.parquet'
FILTERED_DF_PATH = 'filtered.parquet'


def main():
    dataframe = pandas.read_parquet(COMBINED_DF_PATH)

    summaries = dataframe['Summary']

    max_vehicle_severities = get_max_vehicle_severities(dataframe)
    max_occupant_injury_severities = get_max_occupant_injury_severities(dataframe)
    delta_vs = get_delta_vs(dataframe)
    max_injury_counts = get_max_injury_counts(dataframe)

    pandas.DataFrame({
        'Summary': summaries,
        'MaxVehicleSeverity': max_vehicle_severities,
        'MaxOccupantInjurySeverity': max_occupant_injury_severities,
        'DeltaV': delta_vs,
        'MaxInjuryCount': max_injury_counts
    }).to_parquet(FILTERED_DF_PATH)


def get_max_vehicle_severities(dataframe):
    vehicle_severity_mapping = {
        'Unknown': 0,
        'Light ': 1,
        'Moderate ': 2,
        'Severe': 3
    }

    def single_max(vehicles):
        severities = [vehicle['CrashSeverityDesc'] for vehicle in vehicles]
        severities_numeric = [vehicle_severity_mapping[severity] for severity in severities]

        return max(severities_numeric)

    return dataframe['Vehicles'].apply(single_max)


def get_max_occupant_injury_severities(dataframe):
    def single_max(occupants):
        severities = []

        for occupant in occupants:
            max_sev_code = occupant['MaxSevCode']
            injured_status_desc = occupant['InjuredStatusDesc']

            try:
                if 'Not Injured' in injured_status_desc:
                    severities.append(0)
                elif 'Unknown' in injured_status_desc:
                    severities.append(numpy.nan)
                elif int(max_sev_code) > 6:
                    severities.append(numpy.nan)
                else:
                    severities.append(int(max_sev_code))

            except:
                severities.append(numpy.nan)

        return max(severities) if len(severities) > 0 else numpy.nan

    return dataframe['Occupants'].apply(single_max)


def get_delta_vs(dataframe):
    def single_delta_v(vehicles):
        try:
            delta_vs = [vehicle['HighDVDesc'] for vehicle in vehicles]

            delta_vs_numeric = []

            for delta_v in delta_vs:
                try:
                    delta_vs_numeric.append(float(delta_v.replace(' kmph', '')))
                except:
                    delta_vs_numeric.append(numpy.nan)

            delta_vs_numeric = numpy.array(delta_vs_numeric)
            return max(delta_vs_numeric[~numpy.isnan(delta_vs_numeric)])

        except:
            return numpy.nan

    return dataframe['Vehicles'].apply(single_delta_v)


def get_max_injury_counts(dataframe):
    def single_max(occupants):
        injury_counts = []

        for occupant in occupants:
            injury_counts.append(occupant['InjuryCount'])

        return max(injury_counts) if len(injury_counts) > 0 else numpy.nan

    return dataframe['Occupants'].apply(single_max)


if __name__ == '__main__':
    main()
