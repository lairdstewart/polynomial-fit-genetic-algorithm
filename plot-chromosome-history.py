import matplotlib.pyplot as plt
import matplotlib.animation as animation

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                coeffs = [float(x) for x in parts]
                data.append(coeffs)
    return data

def poly_eval(coeffs, x):
    return sum(c * (x ** i) for i, c in enumerate(coeffs))

data = read_data('chromosome-history.dat')
x = [xi / 100 for xi in range(-300, 301)]
true_coeffs = [1, -1, -4, -9, 6.5, 9.5]
truth_x = [xi / 25 for xi in range(-50, 51)]
truth_y = [poly_eval(true_coeffs, xi) for xi in truth_x]

fig, ax = plt.subplots()
line, = ax.plot([], [])
points, = ax.plot([], [], 'ro', markerfacecolor='none', alpha=0.4)
ax.set_xlim(min(x), max(x))
ax.set_ylim(-10, 10)

def animate(i):
    coeffs = data[i]
    y = [poly_eval(coeffs, xi) for xi in x]
    line.set_data(x, y)
    points.set_data(truth_x, truth_y)
    ax.relim()
    ax.autoscale_view()
    ax.set_title(f'Generation {i+1}')
    return line, points, ax.title

ani = animation.FuncAnimation(fig, animate, frames=len(data), interval=200, blit=False)
plt.show()
