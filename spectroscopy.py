import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
from scipy.integrate import simps
from scipy.optimize import curve_fit
import os
style.use('seaborn-dark-palette')


def spectro_all(original, a, b, graph_to, n_line):

    time = datetime.datetime.now()
    directory = os.fsencode(original)
    list_of_powers = np.zeros((len(os.listdir(directory)), 2))
    num = 0
    m_time = f'/measurements{time.strftime("%y%m%d_%H:%M:%S")}'
    g_time = f'/graphs{time.strftime("%y%m%d_%H:%M:%S")}'
    new_directory_m = graph_to + m_time
    new_directory_g = graph_to + g_time
    os.mkdir(new_directory_m)
    os.mkdir(new_directory_g)

    for file in os.listdir(directory):

        file_name = os.fsdecode(file)
        print(file_name)

        with open(original + '/' + file_name, 'r+') as f:
            text = f.read()
            text1 = '\n'.join(text.split('\n')[n_line:])

        with open(new_directory_m + '/' + file_name, 'w+') as f:
            f.write(text1)

        with open(new_directory_m + '/' + file_name, 'r+') as f:
            t = f.read()
            f.seek(0)
            f.truncate()
            f.write(t.replace(',', '.'))

    for file in os.listdir(new_directory_m):

        filename = os.fsdecode(file)
        data = np.loadtxt(new_directory_m + '/' + filename, delimiter='\t')
        print(data.shape)
        print(filename)
        y = []
        x = []

        for i, j in enumerate(data[:, 0]):
            if a < j < b:
                z = data[i, 1]
                t = data[i, 0]
                y.append(z)
                x.append(t)

        my = max(y)
        idx_my = [i for i, j in enumerate(y) if j == my]
        idx_y = int(idx_my[0])

        mean = x[idx_y]

        x = np.array(x)
        y = np.array(y)

        n = len(x)
        miu = max(y)
        sigma = (sum(y * (x - mean) ** 2) / n)**0.5
        dx = n / (b - a)

        def gaussian(x, a, x0, sigma):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

        popt, pcov = curve_fit(gaussian, x, y, p0=[miu, mean, sigma])

        result = gaussian(x, *popt)

        filename = os.fsdecode(file)
        file_name = filename[:-6]
        power = float(file_name)
        path_graph = new_directory_g + f'/{file_name}.png'

        plt.plot(x, y, label=f'{file_name} mW')
        plt.plot(x, result, 'r--', label='fit')
        plt.legend()
        plt.xlabel('wavelength [nm]')
        plt.ylabel('counts')
        plt.savefig(path_graph)
        plt.clf()

        integrate = simps(result, dx=dx)
        list_of_powers[num] = power, integrate
        num += 1

    path_powers = new_directory_m + '/' + f'list_of_powers.txt'
    path_to_pg = new_directory_g + '/' + f'intensity_to_power.png'
    lop = np.sort(list_of_powers.view('i8,i8'), order=['f1'], axis=0).view(np.float)

    x_p = np.array(lop[:, 0])
    y_p = np.array(lop[:, 1])

    plt.scatter(x_p, y_p, label='results')
    plt.legend()
    plt.xlabel('Power [mW]')
    plt.ylabel('Intensity')
    plt.savefig(path_to_pg)
    plt.clf()

    with open(path_powers, 'w')as f:
        for item in list_of_powers:
            f.write(f'{item}\n')

    return list_of_powers
