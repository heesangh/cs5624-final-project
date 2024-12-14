import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt

df = pandas.read_excel('occupants_filtered.xlsx')


def convert_data(value, suffix, data_type):
    try:
        value = value.replace(suffix, '')
        return data_type(value)
    except:
        return numpy.nan


df['age'] = df['AgeInYears'].apply(lambda x: convert_data(x, ' Years', int))
df['height'] = df['Height'].apply(lambda x: convert_data(x, ' cm', int))


def get_injury_severity(row):

    if not numpy.isnan(row['MaxSevCode']):
        return int(row['MaxSevCode'])

    try:
        if row['InjuredStatusDesc'] == 'Not Injured':
            return 0

    except:
        pass

    return numpy.nan


    # try:
    #     if 'Unknown' in row['InjuredStatusDesc']:
    #         return numpy.nan
    #
    #     if row['InjuredStatusDesc'] == 'Not Injured':
    #         return 0
    #
    #     return int(row['MaxSevCode'])
    # except:
    #     return numpy.nan


df['injury_severity'] = df.apply(get_injury_severity, axis=1)

df['weight'] = df['Weight'].apply(lambda x: convert_data(x, ' kgs', int))


def get_crash_time(raw_time):
    try:
        hour = int(raw_time.split(':')[0])
        minute = int(raw_time.split(':')[1])
        return hour * 60 + minute
    except:
        return numpy.nan


df['crash_time'] = df['CrashTime'].apply(get_crash_time)

correlation_matrix = df.corr(numeric_only=True)

injury_correlation = correlation_matrix['injury_severity'].drop('injury_severity').dropna().sort_values(ascending=True)

injury_correlation = injury_correlation.drop('MaxSevCode')
injury_correlation = injury_correlation.drop('InjuredStatus')

plt.figure(figsize=(10, 8))
injury_correlation.plot(kind='barh', color=injury_correlation.apply(lambda x: 'tab:orange' if x < 0 else 'tab:blue'))
plt.title('Feature Correlations with Occupant Injury Severity')
plt.xlabel('Correlation Value')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#
# plt.figure(figsize=(20, 16))
# # plot correlation matrix using seaborn
# seaborn.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.show()
#
# print(correlation_matrix['injury_severity'].sort_values(ascending=False))
