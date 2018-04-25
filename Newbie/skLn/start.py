from skLn.tools import read_csv, norm, separation_data
import numpy as np
from skLn.visual import visual_ln, visual_data
from skLn.learning import sk_learning

place_name = 'kvz'
directory = '/Users/Smoky/Documents/workspace/resourses/csv/sk/' + place_name + 'LN/'
map_size = [36, 52, 37, 46]  # kvz

name_param = {0:'x', 1:'y', 2:'Hmax', 3:'Hmin', 4:'dH', 5:'Hgrad', 6:'Hdisp', 7:'Gmax', 8:'Gmin', 9:'dG', 10:'Gdisp',
              11:'Mmax', 12:'Mmin', 13:'dM', 14:'Mdisp', 15:'Lnint'}

#index_param = [2,3,4,5,7,8,9,15] #standart
index_param = [2,3,4,5,6,7,8,9,15] #standart + disp
#index_param = range(2, 16)
#index_param = [5]



param = [name_param[x] for x in index_param]

# sample, data = norm(sample, data, param_all)

calc_r = lambda x: x / 111



r = calc_r(25)
print(r)
sample = read_csv(directory + place_name + "_eqB.csv", ['x', 'y']).T
sampleX = read_csv(directory + place_name + "_eqXall.csv", ['x', 'y']).T
sampleAll = read_csv(directory + place_name + "_eq6_all.csv", ['x', 'y']).T
data = norm(read_csv(directory + place_name + '_gridv5.csv', ['x', 'y'] + param).T)
# data = read_csv(directory+place_name+'_grid.csv', param_all).T

B, X, H, count = separation_data(data, sample, sampleX, r)
# visual_data(sample, map_size)
print(len(B), 'B')
print(len(X), 'X')
print(len(H), 'H')
print(count, 'count eq ln\n')


#visual_ln(C, None, None, sampleAll, map_size)
sk_learning(B, X, H, sampleAll, map_size, r, param)
