import numpy as np
import openpyxl
import pandas as pd


path = '/Volumes/HDD/data/relief/ETOPO1_Ice_g_int.csv'
array=[]
frame = pd.read_csv(path,header=None, sep=' ', decimal=".",usecols=[0,1,2])
for i in range(3):
    try:
        array.append(frame[i].values)
    except:
        print('error', i)
read_array=np.array(array).T
final_array = []
coordinates = [36, 52, 37.5, 45.5]
#coordinates = [29., 40., 41., 49.] #crimea
#coordinates = [44, 53, 48, 54] #balac
#coordinates = [-127, -113, 30, 42] #calif
#coordinates = [36, 52, 37, 46] #kvz_ln
name = 'kvz_reliefS4'
x_min = coordinates[0]
x_max = coordinates[1]
y_min = coordinates[2]
y_max = coordinates[3]
total = len(read_array)
for num, i in enumerate(read_array):
    if num % 10000000 == 0:
        print(num, 'out of', total)
        print(len(final_array), 'f_array data')
    if (x_min < i[0] and i[0] < x_max) and (y_min < i[1] and i[1]< y_max):
        final_array.append(i)


my_df = pd.DataFrame(np.asarray(final_array), columns=['x', 'y', 'value'])
my_df.to_csv('/Users/Ivan/Documents/workspace/resourses/csv/geop/'+name+'.csv', index=False, header=True,
                 sep=';', decimal=',')