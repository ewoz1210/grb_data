import pandas as pd
import numpy as np
from astropy.time import Time
from astropy.cosmology import WMAP9 as cosmo
import astropy.coordinates as coord
import astropy.units as u
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import sys

# filters out warnings that may arise with dates
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)


# read and store content of csv's in data frames
df1 = pd.read_csv('data/batsegrb.csv')
df2 = pd.read_csv('data/fermigbrst.csv')
df3 = pd.read_csv('data/fermilpsc.csv')
df4 = pd.read_csv('data/grbcat.csv')
df5 = pd.read_csv('data/saxgrbmgrb.csv')
df6 = pd.read_csv('data/swiftgrb.csv')

# list of all data frames
frames = [df1, df2, df3, df4, df5, df6]

# convert df6 time_x to mjd and add to a new column, and delete old time column
conv_time = []
for t in df6['time_x']:
    t_new = (Time(t)).mjd
    conv_time.append(t_new)
se = pd.Series(conv_time)
df6['time'] = se.values
del df6['time_x']

# concats all data frames together and rounds time column to 2 decimals
df_all = pd.concat(frames)
df_all.time = df_all.time.round(2)
# sort descending according to redshift, fluence, ra, dec
df_all = (df_all.sort_values(['redshift', 'fluence', 'ra', 'dec'], ascending=False))
# removes duplicate entries, based on both time and name keeping first
df_all = df_all.drop_duplicates(subset=['name', 'time'], keep='first')
# sorts based on time and resets index
df_all = (df_all.sort_values(['time'])).reset_index(drop=True)

print ('\nAt any prompt, enter Q to exit the program.')

# User setting date range
# sets default range or user defined range
# input must be in proper format, must be between default range, and end date must be after start date
while True:
    print ('\nDefault date range is from 1967-07-02 to 2016-11-19.\nSelect default range or choose two dates within that range.')
    start_d_raw = input('\nEnter start date or press ENTER for default (match format: yyyy-mm-dd) > ')
    if start_d_raw.lower() == 'q':
        sys.exit()
    else:
        pass
    end_d_raw = input('Enter end date or press ENTER for default (match format: yyyy-mm-dd) > ')
    if end_d_raw.lower() == 'q':
        sys.exit()
    else:
        pass
    if start_d_raw == '' and end_d_raw == '':
        start_d = 39673.6
        end_d = 57711.63
        break
    elif start_d_raw == '':
        start_d = 39673.6
        try:
            end_d = (Time(end_d_raw)).mjd
            if end_d >= start_d:
                if end_d <= 57711.63:
                    break
                else:
                    print ('Dates must be within default range.')
                    continue
            else:
                print ('End date must be at least one day after start date.')
                continue
        except ValueError:
            print ('Please enter a date in the correct format.')
            continue
    elif end_d_raw == '':
        end_d = 57711.63
        try:
            start_d = (Time(start_d_raw)).mjd
            if end_d >= start_d:
                if start_d >= 39673.6:
                    break
                else:
                    print ('Dates must be within default range.')
                    continue
            else:
                print ('End date must be after start date.')
                continue
        except ValueError:
            print ('Please enter a date in the correct time range and format.')
            continue
    else:
        try:
            start_d = (Time(start_d_raw)).mjd
            if start_d >= 39673.6:
                pass
            else:
                print ('Dates must be within default range.')
                continue
        except ValueError:
            print ('Please enter a date in the correct format.')
            continue
        try:
            end_d = (Time(end_d_raw)).mjd
            if end_d >= start_d:
                if end_d <= 57711.63:
                    break
                else:
                    print ('End date must be after start date.')
                    continue
            else:
                print ('End date must be after start date.')
                continue
        except ValueError:
            print ('Please enter a date in the correct format.')
            continue

# convert dates to iso format for display
start_date = (Time(start_d, format='mjd', scale='utc')).iso
end_date = (Time(end_d, format='mjd', scale='utc')).iso
print ('\nStart date of search:',start_date)
print ('\nEnd date of search:',end_date)

# now use date selection to create the data frame to be used for functions below
df_sel = df_all[df_all['time'].between(start_d, end_d, inclusive=True)]

#DATES
# What is date of the earliest recording?
def earliest():
    t = ((df_sel.sort_values(['time']).reset_index(drop=True))['time'][0])
    t = (Time(t, format='mjd', scale='utc')).iso
    return t

# What is date of the latest recording?
def latest():
    t = ((df_sel.sort_values(['time'], ascending=False).reset_index(drop=True))['time'][0])
    t = (Time(t, format='mjd', scale='utc')).iso
    return t

# What is the difference in days between earliest and latest?
def time_delta():
    date_format = "%Y-%m-%d %H:%M:%S.%f"
    tda = datetime.strptime(start_date, date_format)
    tdb = datetime.strptime(end_date, date_format)
    td = (tdb - tda).days
    return td

#GRB's
# How many gamma-ray bursts have been recorded?
def num_grbs():
    g = len(df_sel)
    return g

# What is the average number of gamma-ray bursts recorded?
def avg_num_grbs():
    g = num_grbs()
    td = time_delta()
    avg_num_g = np.round((g / td), decimals=3)
    return avg_num_g

#FLUENCE
# How many recordings of fluence total in the database?
def num_fluence():
    return df_sel['fluence'].count()

# What is the average number of fluences recorded?
def avg_num_fluence():
    td = time_delta()
    avg_num_f = np.round((num_fluence() / td), decimals=3)
    return avg_num_f

# What is the largest fluence recorded?
def gr_fluence():
    f = ((df_sel.sort_values(['fluence'], ascending=False).reset_index(drop=True))['fluence'][0])
    return f

#REDSHIFT
# How many recordings of redshifts total in the database?
def num_redshift():
    return df_sel['redshift'].count()

# What is the average number of redshifts recorded?
def avg_num_redshift():
    td = time_delta()
    avg_num_r = np.round((num_redshift() / td), decimals=3)
    return avg_num_r

# What is the largest redshift measured?
def gr_redshift():
    z = ((df_sel.sort_values(['redshift'], ascending=False).reset_index(drop=True))['redshift'][0])
    return z

# How old was the universe at the time of furthest grb?
def univ_age():
    z = ((df_sel.sort_values(['redshift'], ascending=False).reset_index(drop=True))['redshift'][0])
    u_age = (cosmo.age(z).value)*1000
    u_age = np.round(u_age, decimals=4)
    return u_age


# Which outputs does user want?
while True:
    print ("\nView GRB(G), fluence(F), or redshift(R) data. Or view available plots(P).")
    data_choice = input("Enter G, F, R, P > ")
    if data_choice.lower() == 'g':
        print ('\nEarliest record:',earliest())
        print ('\nLatest record:',latest())
        print ('\nNumber of days in selected range:',time_delta(),'days')
        print ('\nNumber of gamma-ray bursts recorded:',num_grbs())
        print ('\nAverage number of gamma-ray bursts recorded:',avg_num_grbs(),'per day')
    elif data_choice.lower() == 'f':
        print ('\nEarliest record:',earliest())
        print ('\nLatest record:',latest())
        print ('\nNumber of days in selected range:',time_delta(),'days')
        print ('\nTotal number of fluences recorded:',num_fluence())
        print ('\nAverage number of fluences recorded:',avg_num_fluence(),'per day')
        print ('\nLargest fluence recorded:',gr_fluence(),'erg/cm^2')
    elif data_choice.lower() == 'r':
        print ('\nEarliest record:',earliest())
        print ('\nLatest record:',latest())
        print ('\nNumber of days in selected range:',time_delta(),'days')
        print ('\nTotal number of redshifts recorded:',num_redshift())
        print ('\nAverage number of redshifts recorded:',avg_num_redshift(),'per day')
        print ('\nLargest redshift recorded:',gr_redshift())
        print ('\nThe GRB with the redshift of',gr_redshift(),'occurred when the universe was',univ_age(),'million years old.')
    elif data_choice.lower() == 'q':
        sys.exit()
    elif data_choice.lower() == 'p':
        plt.style.use('ggplot')
        print ('\nAvailable plots include: Location(L), Fluence v. Redshift(FR), or Redshift v. Age of Universe (RA)')
        plot_choice = input('Enter L, FR, or RA > ')
        if plot_choice.lower() == 'l':
            ra = coord.Angle(df_sel['ra']*u.degree)
            ra = ra.wrap_at(180*u.degree)
            dec = coord.Angle(df_sel['dec']*u.degree)
            fig = plt.figure(figsize=(14,8))
            ax = fig.add_subplot(111, projection="mollweide")
            ax.scatter(ra.radian, dec.radian, color='#52CEFF')
            ax.grid(True)
            plt.title("Gamma-ray burst locations", fontsize=36)
            plt.xlabel("Right Ascension(RA)", fontsize=24)
            plt.ylabel("Declination(dec)", fontsize=24)
            plt.show()
            continue
        elif plot_choice.lower() == 'fr':
            red = df6['redshift']
            flu = df6['fluence']
            fig = plt.figure(figsize=(14,8))
            ax = fig.add_subplot(111)
            ax.scatter(red, flu, color='DarkBlue')
            ax.grid(True)
            plt.title("Modeling GRB's observed by Swift", fontsize=36)
            plt.xlabel("Redshift", fontsize=24)
            plt.ylabel("Fluence(erg/cm^2)", fontsize=24)
            plt.ylim([0,.00005])
            plt.xlim([0,12])
            plt.show()
            continue
        elif plot_choice.lower() == 'ra':
            z = df6['redshift']
            u_age = (cosmo.age(z).value)*1000
            fig = plt.figure(figsize=(14,8))
            ax = fig.add_subplot(111)
            ax.scatter(u_age, z, color='#CC0000')
            ax.grid(True)
            ax.annotate('Big Bang', fontsize=20, xy=(0, 0), xytext=(20, 2),
            arrowprops=dict(facecolor='black'),
            )
            plt.title("GRB Redshifts and Age of the Universe", fontsize=36)
            plt.xlabel("Age of Universe(in millions of years)", fontsize=24)
            plt.ylabel("Redshift", fontsize=24)
            plt.ylim([0,12])
            plt.xlim([0,2000])
            plt.show()
            continue
        elif plot_choice == 'q':
            sys.exit()
        else:
            continue
    else:
        continue
