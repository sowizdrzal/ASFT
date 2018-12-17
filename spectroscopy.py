import numpy as np
import shutil
import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
from scipy.integrate import simps
from scipy.optimize import curve_fit
import os
style.use('seaborn-dark-palette')


def copy_original(source, destination):

    files = os.listdir(source)

    for f in files:
        shutil.copy(source + f, destination)


def preprocessing(dire, n_line):

    directory = os.fsencode(dire)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)
        path_to = dire + '/' + filename
        n_firstlines = []

        with open(path_to, 'r+') as f:
            text = f.read()
            f.seek(0)
            f.truncate()
            f.write(text.replace(',', '.'))

        with open(path_to) as f, open("test.txt", "w") as out:
            for x in range(n_line):
                n_firstlines.append(next(f))
            for line in f:
                out.write(line)
        os.remove(path_to)
        os.rename('test.txt', path_to)


def spectro_all(dire, a, b, graph_to):

    time = datetime.datetime.now()
    directory = os.fsencode(dire)
    list_of_powers = np.zeros((len(os.listdir(directory)), 2))
    num = 0
    new_directory = graph_to + '/' + f'wyniki{time.strftime("%y%m%d_%H:%M")}'
    new_directory_1 = graph_to + '/' + f'wykresy{time.strftime("%y%m%d_%H:%M")}'
    os.mkdir(new_directory)
    os.mkdir(new_directory_1)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)
        path_to = dire + '/' + filename

        data = np.loadtxt(path_to, delimiter="\t")

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

        file_name = filename[:-6]
        power = float(file_name)
        path_graph = new_directory_1 + '/' + f'{file_name}.png'
        path_to_new = new_directory + '/' + f'{filename}'

        plt.plot(x, y, label=f'{file_name} mW')
        plt.plot(x, result, 'r--', label='fit')
        plt.legend()
        plt.xlabel('wavelength [nm]')
        plt.ylabel('counts')
        plt.savefig(path_graph)
        plt.clf()

        integrate = simps(result, dx=dx)
        os.rename(path_to, path_to_new)
        list_of_powers[num] = power, integrate
        num += 1

    path_powers = new_directory + '/' + f'list_of_powers.txt'
    path_to_pg = new_directory_1 + '/' + f'intensity_to_power.png'
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





