import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import scipy
import numpy as np
import keras
import tensorflow as tf
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import FeatureHasher
from collections import Counter



#----------------------- CREATE THE DATAFRAME ------------------------#
df = pd.read_csv('hate_crime_data 2.csv')
# The below comments are useful for debugging / visualizing
#print(df)
#print(df.shape)
#print(df.columns)
#print(df.isnull().sum())
#print(df.info())
#print(df.describe())
#print(df.head(10))
'''
print(df.shape)
#print(df.head(10))
df.to_csv('output_file.csv', index=False)
column = df['victim_types']
counter_summary = Counter(column)
print(counter_summary)
'''
#----------------------- CREATE THE DATAFRAME ------------------------#

#----------------- DROP OLD DATA & UNNEEDED COLUMNS ------------------#
df = df[df['data_year'] >= 2000]
print(df.isnull().sum())
df = df.drop(['pug_agency_name',
              'pub_agency_unit', 
              'agency_type_name', 
              'state_abbr', 
              'incident_id', 
              'ori',
              'offender_race',
              'offender_ethnicity',
              'population_group_code',
              'division_name',
              'region_name'], axis=1)
df = df.dropna(subset=['population_group_description'])
#----------------- DROP OLD DATA & UNNEEDED COLUMNS ------------------#

#--------------- ENCODE DATE INTO NUMERICAL CATEGORIES ---------------#
df['incident_year'] = pd.to_datetime(df['incident_date']).dt.year
df['incident_month'] = pd.to_datetime(df['incident_date']).dt.month
df['incident_day'] = pd.to_datetime(df['incident_date']).dt.day
#--------------- ENCODE DATE INTO NUMERICAL CATEGORIES ---------------#

#------------------ ENCODE MUTIPLE OFFENSE AND BIAS ------------------#
df['multiple_offense'] = df['multiple_offense'].map({'M': 1, 'S': 0})
df['multiple_bias'] = df['multiple_bias'].map({'M': 1, 'S': 0})

# Map 'M' to 1 and 'S' to 0 in multiple offense and bias columns
df['multiple_offense'] = df['multiple_offense'].map({'M': 1, 'S': 0})
df['multiple_bias'] = df['multiple_bias'].map({'M': 1, 'S': 0})
#------------------ ENCODE MUTIPLE OFFENSE AND BIAS ------------------#

#------------------- ENCODE THE POPULATION GROUPS --------------------#
def classify_population_group(category):
    if '1,000,000 or over' in category or '500,000 thru 999,999' in category:
        return 'Large_City'
    elif '100,000 thru 249,999' in category or '250,000 thru 499,999' in category:
        return 'Medium_City'
    elif '10,000 thru 24,999' in category or '25,000 thru 49,999' in category:
        return 'Small_City'
    elif 'under 10,000' in category or '2,500 thru 9,999' in category:
        return 'Rural'
    else:
        return 'Other'

df['population_group_class'] = df['population_group_description'].apply(classify_population_group)
df = pd.get_dummies(df, columns=['population_group_class'])
for col in df.filter(like='population_group_class_').columns:
    df[col] = df[col].astype(int)
#------------------- ENCODE THE POPULATION GROUPS --------------------#

#------------------------ ENCODE VICTIM TYPES ------------------------#
df = df.dropna(subset=['victim_types'])

def classify_victim_type(victim):
    components = victim.split(';')
    if len(components) == 1:
        return components[0]
    elif 'Unknown' in components:
        return 'Unknown'
    else:
        return 'Mixed_Entity'
    
df['victim_type_class'] = df['victim_types'].apply(classify_victim_type)
df = pd.get_dummies(df, columns=['victim_type_class'])
for col in df.filter(like='victim_type_class_').columns:
    df[col] = df[col].astype(int)
#------------------------ ENCODE VICTIM TYPES ------------------------#

#-------------------------- ENCODING STATES --------------------------#
le = LabelEncoder()
df['state_encoded'] = le.fit_transform(df['state_name'])
#-------------------------- ENCODING STATES --------------------------#

#----------- FILLING NAN VALUES WITH 0 FOR NUMERIC COLUMNS -----------#
numeric_columns = ['adult_victim_count', 'juvenile_victim_count', 'adult_offender_count', 'juvenile_offender_count']
df[numeric_columns] = df[numeric_columns].fillna(0)
#----------- FILLING NAN VALUES WITH 0 FOR NUMERIC COLUMNS -----------#

#----------------------GROUP OFFENSE CATEGORIES ----------------------#
offense_category_map = {
    'Aggravated Assault': 'offense_name_ViolentCrime',
    'All Other Larceny': 'offense_name_PropertyCrime',
    'Animal Cruelty': 'offense_name_OtherCrime',
    'Arson': 'offense_name_PropertyCrime',
    'Assisting or Promoting Prostitution': 'offense_name_SexCrime',
    'Betting/Wagering': 'offense_name_Gambling',
    'Bribery': 'offense_name_OtherCrime',
    'Burglary/Breaking & Entering': 'offense_name_PropertyCrime',
    'Counterfeiting/Forgery': 'offense_name_Fraud',
    'Credit Card/Automated Teller Machine Fraud': 'offense_name_Fraud',
    'Destruction/Damage/Vandalism of Property': 'offense_name_PropertyCrime',
    'Drug Equipment Violations': 'offense_name_DrugCrime',
    'Drug/Narcotic Violations': 'offense_name_DrugCrime',
    'Embezzlement': 'offense_name_Fraud',
    'Extortion/Blackmail': 'offense_name_Fraud',
    'False Pretenses/Swindle/Confidence Game': 'offense_name_Fraud',
    'Fondling': 'offense_name_ViolentCrime',
    'Gambling Equipment Violation': 'offense_name_Gambling',
    'Hacking/Computer Invasion': 'offense_name_CyberCrime',
    'Human Trafficking, Commercial Sex Acts': 'offense_name_SexCrime',
    'Human Trafficking, Involuntary Servitude': 'offense_name_SexCrime',
    'Identity Theft': 'offense_name_Fraud',
    'Impersonation': 'offense_name_Fraud',
    'Incest': 'offense_name_SexCrime',
    'Intimidation': 'offense_name_ViolentCrime',
    'Kidnapping/Abduction': 'offense_name_ViolentCrime',
    'Murder and Nonnegligent Manslaughter': 'offense_name_ViolentCrime',
    'Motor Vehicle Theft': 'offense_name_PropertyCrime',
    'Negligent Manslaughter': 'offense_name_ViolentCrime',
    'Not Specified': 'offense_name_OtherCrime',
    'Operating/Promoting/Assisting Gambling': 'offense_name_Gambling',
    'Pocket-picking': 'offense_name_PropertyCrime',
    'Pornography/Obscene Material': 'offense_name_SexCrime',
    'Prostitution': 'offense_name_SexCrime',
    'Purchasing Prostitution': 'offense_name_SexCrime',
    'Purse-snatching': 'offense_name_PropertyCrime',
    'Rape': 'offense_name_SexCrime',
    'Robbery': 'offense_name_ViolentCrime',
    'Sexual Assault With An Object': 'offense_name_SexCrime',
    'Shoplifting': 'offense_name_PropertyCrime',
    'Simple Assault': 'offense_name_ViolentCrime',
    'Sodomy': 'offense_name_SexCrime',
    'Stolen Property Offenses': 'offense_name_PropertyCrime',
    'Statutory Rape': 'offense_name_SexCrime',
    'Theft From Building': 'offense_name_PropertyCrime',
    'Theft From Coin-Operated Machine or Device': 'offense_name_PropertyCrime',
    'Theft From Motor Vehicle': 'offense_name_PropertyCrime',
    'Theft of Motor Vehicle Parts or Accessories': 'offense_name_PropertyCrime',
    'Weapon Law Violations': 'offense_name_WeaponCrime',
    'Wire Fraud': 'offense_name_Fraud',
    'Welfare Fraud': 'offense_name_Fraud',
    'Federal Liquor Offenses': 'offense_name_OtherCrime',
}

def map_offenses_to_categories(offenses):
    offense_list = offenses.split(';')
    categories = set()
    for offense in offense_list:
        offense = offense.strip()
        category = offense_category_map.get(offense, 'Other Crime')
        categories.add(category)
    return list(categories)

df['offense_categories'] = df['offense_name'].apply(map_offenses_to_categories)

mlb = MultiLabelBinarizer()
cols = mlb.fit_transform(df['offense_categories'])

temp = pd.DataFrame(cols, columns=mlb.classes_, index=df.index)
df = pd.concat([df, temp], axis=1)
#----------------------GROUP OFFENSE CATEGORIES ----------------------#

#------------------ GROUP LOCATION NAME CATEGORIES -------------------#
location_category_map = {
    'Residence/Home': 'location_name_Residential',
    'Hotel/Motel/Etc.': 'location_name_Residential',
    'Shelter-Mission/Homeless': 'location_name_Residential',
    'Abandoned/Condemned Structure': 'location_name_Residential',
    'Tribal Lands': 'location_name_Residential',
    'Daycare Facility': 'location_name_Educational',
    'Construction Site': 'location_name_Industrial',
    'Industrial Site': 'location_name_Industrial',
    'Farm Facility': 'location_name_Agricultural',
    'Camp/Campground': 'location_name_Recreational',
    'Highway/Road/Alley/Street/Sidewalk': 'location_name_Transportation',
    'Parking/Drop Lot/Garage': 'location_name_Transportation',
    'Air/Bus/Train Terminal': 'location_name_Transportation',
    'Dock/Wharf/Freight/Modal Terminal': 'location_name_Transportation',
    'Rest Area': 'location_name_Transportation',
    'Restaurant': 'location_name_Commercial',
    'Bar/Nightclub': 'location_name_Commercial',
    'Convenience Store': 'location_name_Commercial',
    'Commercial/Office Building': 'location_name_Commercial',
    'Specialty Store': 'location_name_Commercial',
    'Service/Gas Station': 'location_name_Commercial',
    'Department/Discount Store': 'location_name_Commercial',
    'Grocery/Supermarket': 'location_name_Commercial',
    'Bank/Savings and Loan': 'location_name_Commercial',
    'Liquor Store': 'location_name_Commercial',
    'Shopping Mall': 'location_name_Commercial',
    'Auto Dealership New/Used': 'location_name_Commercial',
    "Drug Store/Doctor's Office/Hospital": 'location_name_Medical',
    'Arena/Stadium/Fairgrounds/Coliseum': 'location_name_Recreational',
    'Gambling Facility/Casino/Race Track': 'location_name_Recreational',
    'Amusement Park': 'location_name_Recreational',
    'Rental Storage Facility': 'location_name_Commercial',
    'Government/Public Building': 'location_name_Government',
    'Jail/Prison/Penitentiary/Corrections Facility': 'location_name_Government',
    'Military Installation': 'location_name_Government',
    'School/College': 'location_name_Educational',
    'School-Elementary/Secondary': 'location_name_Educational',
    'School-College/University': 'location_name_Educational',
    'Park/Playground': 'location_name_Recreational',
    'Field/Woods': 'location_name_Recreational',
    'Lake/Waterway/Beach': 'location_name_Recreational',
    'Church/Synagogue/Temple/Mosque': 'location_name_Religious',
    'Cyberspace': 'location_name_Cyberspace',
    'Community Center': 'location_name_Community',
    'ATM Separate from Bank': 'location_name_Commercial',

}

def map_locations_to_categories(location_str):
    location_list = location_str.split(';')
    categories = set()
    for location in location_list:
        location = location.strip()
        category = location_category_map.get(location, 'location_name_Other')
        categories.add(category)
    return list(categories)

df['location_categories'] = df['location_name'].apply(map_locations_to_categories)

mlb = MultiLabelBinarizer()

cols = mlb.fit_transform(df['location_categories'])
temp = pd.DataFrame(cols, columns=mlb.classes_, index=df.index)
df = pd.concat([df, temp], axis=1)
#------------------ GROUP LOCATION NAME CATEGORIES -------------------#

#----------------- GROUP BIAS DESCRIPTION CATEGORIES -----------------#
bias_category_map = {
    # Race/Ethnicity/Ancestry Biases
    'Anti-Black or African American': 'bias_desc_Race',
    'Anti-White': 'bias_desc_Race',
    'Anti-Asian': 'bias_desc_Race',
    'Anti-Arab': 'bias_desc_Race',
    'Anti-Hispanic or Latino': 'bias_desc_Race',
    'Anti-American Indian or Alaska Native': 'bias_desc_Race',
    'Anti-Native Hawaiian or Other Pacific Islander': 'bias_desc_Race',
    'Anti-Multiple Races, Group': 'bias_desc_Race',
    'Anti-Other Race/Ethnicity/Ancestry': 'bias_desc_Race',

    # Religion Biases
    'Anti-Jewish': 'bias_desc_Religion',
    'Anti-Islamic (Muslim)': 'bias_desc_Religion',
    'Anti-Catholic': 'bias_desc_Religion',
    'Anti-Protestant': 'bias_desc_Religion',
    'Anti-Other Religion': 'bias_desc_Religion',
    'Anti-Multiple Religions, Group': 'bias_desc_Religion',
    'Anti-Atheism/Agnosticism': 'bias_desc_Religion',
    'Anti-Eastern Orthodox (Russian, Greek, Other)': 'bias_desc_Religion',
    'Anti-Hindu': 'bias_desc_Religion',
    'Anti-Sikh': 'bias_desc_Religion',
    "Anti-Jehovah's Witness": 'bias_desc_Religion',
    'Anti-Other Christian': 'bias_desc_Religion',
    'Anti-Church of Jesus Christ': 'bias_desc_Religion',

    # Sexual Orientation Biases
    'Anti-Gay (Male)': 'bias_desc_Sexual_Orientation',
    'Anti-Lesbian (Female)': 'bias_desc_Sexual_Orientation',
    'Anti-Lesbian, Gay, Bisexual, or Transgender (Mixed Group)': 'bias_desc_Sexual_Orientation',
    'Anti-Bisexual': 'bias_desc_Sexual_Orientation',
    'Anti-Heterosexual': 'bias_desc_Sexual_Orientation',

    # Gender Biases
    'Anti-Male': 'bias_desc_Gender',
    'Anti-Female': 'bias_desc_Gender',

    # Gender Identity Biases
    'Anti-Transgender': 'bias_desc_Gender_Identity',
    'Anti-Gender Non-Conforming': 'bias_desc_Gender_Identity',

    # Disability Biases
    'Anti-Physical Disability': 'bias_desc_Disability',
    'Anti-Mental Disability': 'bias_desc_Disability',

    # Other
    "Unknown (offender's motivation not known)": 'bias_desc_Other',
    # Any biases not explicitly mapped will be categorized as 'bias_desc_Other'
}

def map_biases_to_categories(bias_str):
    bias_list = bias_str.split(';')
    categories = set()
    for bias in bias_list:
        bias = bias.strip()
        category = bias_category_map.get(bias, 'bias_desc_Other')
        categories.add(category)
    return list(categories)


df['bias_categories'] = df['bias_desc'].apply(map_biases_to_categories)
mlb = MultiLabelBinarizer()

bias_dummies = mlb.fit_transform(df['bias_categories'])
bias_dummies_df = pd.DataFrame(bias_dummies, columns=mlb.classes_, index=df.index)
df = pd.concat([df, bias_dummies_df], axis=1)
#----------------- GROUP BIAS DESCRIPTION CATEGORIES -----------------#

#---------------- DROP COLUMNS THAT HAVE BEEN ENCODED ----------------#
df = df.drop('data_year', axis=1)
df = df.drop('victim_types', axis=1)
df = df.drop('population_group_description', axis=1)
df = df.drop('state_name', axis=1)
df = df.drop('incident_date', axis=1)
df = df.drop(['offense_name', 'offense_categories', 'multiple_offense'], axis=1)
df = df.drop(['location_name', 'location_categories', 'multiple_bias'], axis=1)
df = df.drop(['bias_desc', 'bias_categories'], axis=1)
#---------------- DROP COLUMNS THAT HAVE BEEN ENCODED ----------------#

#------------------------------ IMPUTING -----------------------------#
# Print statements useful for debugging
# print("Missing values before imputation:")
# print(df.isnull().sum())

df = df.fillna(0)

# print("\nMissing values after imputation:")
# print(df.isnull().sum())
#------------------------------ IMPUTING -----------------------------#

#--------------------------- VISUALIZATIONS --------------------------#
corr_matrix = df.corr(method='pearson')
plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
ticks = np.arange(0, len(corr_matrix.columns), 1)
plt.xticks(ticks, corr_matrix.columns, rotation=90)
plt.yticks(ticks, corr_matrix.columns)
plt.title('Pearson Correlation Matrix')
plt.tight_layout()
plt.show()

df.to_csv('processed_data.csv', index=False)
#--------------------------- VISUALIZATIONS --------------------------#