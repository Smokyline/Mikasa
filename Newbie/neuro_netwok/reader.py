import openpyxl
import pandas as pd
import numpy as np


def read(path):
    data = []
    wb = openpyxl.load_workbook(path)
    sheet = wb.get_sheet_by_name(wb.get_sheet_names()[0])
    for row in sheet:
        rowData = []
        for cell in row:
            rowData.append(cell.value)
        data.append(rowData)
    print('done')
    return data

def read_csv(path, param):
    array=[]
    frame = pd.read_csv(path, header=0, sep=';',decimal=",")
    for i,title in enumerate( np.append(['x','y'],param)):
        try:
            array.append(frame[title].values)
        except:
            print('no_'+title, end=' ')

    return np.array(array)