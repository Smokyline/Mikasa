
from skMLA.learning import sk_learn
from skLn.tools import norm
from skMLA.tools import read_csv, normal

place_name = 'kvz'
directory = '/Users/Smoky/Documents/workspace/resourses/csv/sk/' + place_name + 'MLA/'
map_size = [40, 54, 35, 47]  # kvz
# map_size = [29., 40., 41., 49.] #crimea

name_param = {0: 'x', 1: 'y', 2: 'Hmax', 3: 'Hmin', 4: 'dH', 5: 'Hgrad', 6: 'Hdisp', 7: 'Gmax', 8: 'Gmin', 9: 'dG', 10: 'Ggrad',
              11: 'Gdisp', 12: 'Mmax', 13: 'Mmin', 14: 'dM', 15: 'Mgrad', 16: 'Mdisp', 17: 'Lnint', 18: 'DPSc', 19: 'DPSint'}
index_param = range(2, 20)
#index_param = [2,3,8,10,11,12,13,15,16,17,18,19]

param = [name_param[x] for x in index_param]

# sample, data = norm(sample, data, param_all)

calc_r = lambda x: x / 111


sampleAll = read_csv(directory + place_name + "_eq55-6.csv", ['x', 'y']).T
#grid = norm(read_csv(directory + place_name + '_grid.csv', param_all).T)
B = read_csv(directory + place_name + "_gridB.csv", ['x', 'y']+param).T
X = read_csv(directory + place_name + "_gridX.csv", ['x', 'y']+param).T
H = read_csv(directory + place_name + "_gridH.csv", ['x', 'y']+param).T
B, X, H = normal(B, X, H)
# data = read_csv(directory+place_name+'_grid.csv', param_all).T

#B, X, H, C = separation_data(data, sampleB, sampleX, r)
print(len(B), 'B')
print(len(X), 'X')
print(len(H), 'H')
sk_learn(B, X, H, param, sampleAll, map_size)
