import numpy as np




size_axis = (3, 3)
center_w_l = (1, 1)


def create_axis_indexes(size_axis, center_w_l):
    coordinates = []
    for i in range(-center_w_l, size_axis - center_w_l):
        coordinates.append(i)
    return coordinates


def create_indexes(size_axis, center_w_l):
    # расчет координат на осях ядра свертки в зависимости от номера центрального элемента ядра
    axis_x = create_axis_indexes(size_axis=size_axis[0], center_w_l=center_w_l[0])
    axis_y = create_axis_indexes(size_axis=size_axis[1], center_w_l=center_w_l[1])
    return axis_x, axis_y


print(create_indexes(size_axis, center_w_l))
