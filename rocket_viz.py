# 3D Rocket Orientation + Flight Data Visualization
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from inertial_vs_barom import local_to_global_ac, quat_to_matrix

TIME_START = 575
TIME_END = 610


def integrate(time_data, data):
    result = [0.0]
    for i in range(1, len(time_data)):
        dt = time_data[i] - time_data[i - 1]
        avg = (data[i] + data[i - 1]) / 2.0
        result.append(result[-1] + avg * dt)
    return np.array(result)


def load_data():
    time, quats, altitudes, accels = [], [], [], []
    with open('LOG028.TXT', 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            v = row['Time'].split(',')
            t = float(v[0])
            # Filter by configurable time range (convert seconds to ms)
            if t < TIME_START * 1000 or t > TIME_END * 1000:
                continue
            xg, yg, zg = map(float, v[1:4])
            alt = float(v[9])
            bno_i, bno_j, bno_k, bno_real = map(float, v[13:17])
            time.append(t / 1000)  # ms to seconds
            quats.append((bno_real, bno_i, bno_j, bno_k))
            altitudes.append(alt)
            accels.append((xg, yg, zg))

    # Compute global Z accel and inertial altitude
    global_z = []
    body_y = []
    for idx, acc in enumerate(accels):
        # quats stored as (w, x, y, z) = (real, i, j, k)
        w, x, y, z = quats[idx]
        global_acc = local_to_global_ac(acc, (w, x, y, z))
        global_z.append(global_acc[2])
        body_y.append(acc[1])

    # Inertial altitude from body Y (thrust axis) directly
    body_y = np.array(body_y)
    global_z = np.array(global_z)
    vertical_acc_y = (body_y - 1.0) * 9.80665  # g to m/s^2
    velocity = integrate(time, vertical_acc_y)
    inertial_alt = integrate(time, velocity)

    # Subsample for animation
    step = 3
    return (np.array(time)[::step],
            [quats[i] for i in range(0, len(quats), step)],
            np.array(altitudes)[::step],
            np.array(body_y)[::step],
            np.array(global_z)[::step],
            np.array(inertial_alt)[::step])


def draw_rocket(ax, R):
    ax.clear()
    n = 12
    r, h = 0.15, 1.0
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)

    bottom = [(r*np.cos(t), 0, r*np.sin(t)) for t in theta]
    top = [(r*np.cos(t), h, r*np.sin(t)) for t in theta]
    tip = (0, h + 0.4, 0)

    bottom = [R @ np.array(p) for p in bottom]
    top = [R @ np.array(p) for p in top]
    tip = R @ np.array(tip)

    for i in range(n):
        j = (i + 1) % n
        ax.add_collection3d(Poly3DCollection(
            [[bottom[i], bottom[j], top[j], top[i]]],
            alpha=0.7, facecolor='silver', edgecolor='gray'))

    for i in range(n):
        j = (i + 1) % n
        ax.add_collection3d(Poly3DCollection(
            [[top[i], top[j], tip]],
            alpha=0.8, facecolor='red', edgecolor='darkred'))

    # World axes
    ax.quiver(0, 0, 0, 1.5, 0, 0, color='r', alpha=0.5)
    ax.quiver(0, 0, 0, 0, 1.5, 0, color='g', alpha=0.5)
    ax.quiver(0, 0, 0, 0, 0, 1.5, color='b', alpha=0.5)
    ax.text(1.6, 0, 0, 'X', color='r')
    ax.text(0, 1.6, 0, 'Y↑', color='g')
    ax.text(0, 0, 1.6, 'Z', color='b')

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def main():
    print("Loading data...")
    time, quats, baro_alt, body_y, global_z, inertial_alt = load_data()
    print(f"Loaded {len(time)} samples")

    fig = plt.figure(figsize=(16, 10))

    # 3D rocket (left half)
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')

    # Barometric altitude (top right)
    ax_baro = fig.add_subplot(2, 2, 2)
    ax_baro.plot(time, baro_alt, 'g-', linewidth=1)
    ax_baro.set_ylabel('Altitude (m)')
    ax_baro.set_title('Barometric Altitude')
    ax_baro.grid(True, alpha=0.3)
    marker_baro, = ax_baro.plot([time[0]], [baro_alt[0]], 'ro', markersize=8)

    # Inertial altitude from body Y (bottom left)
    ax_inert = fig.add_subplot(2, 2, 3)
    ax_inert.plot(time, inertial_alt, 'r-', linewidth=1)
    ax_inert.set_ylabel('Altitude (m)')
    ax_inert.set_title('Inertial Altitude (from Body Y)')
    ax_inert.grid(True, alpha=0.3)
    marker_inert, = ax_inert.plot(
        [time[0]], [inertial_alt[0]], 'ro', markersize=8)

    # Accelerations (bottom right)
    ax_acc = fig.add_subplot(2, 2, 4)
    ax_acc.plot(time, body_y, 'b-', linewidth=0.8,
                alpha=0.7, label='Body Y (thrust axis)')
    ax_acc.plot(time, global_z, 'orange', linewidth=0.8,
                alpha=0.7, label='Global Z (vertical)')
    ax_acc.axhline(y=1.0, color='gray', linestyle='--',
                   alpha=0.5, label='1g reference')
    ax_acc.set_xlabel('Time (s)')
    ax_acc.set_ylabel('Acceleration (g)')
    ax_acc.set_title('Body Y vs Global Z Acceleration')
    ax_acc.legend(loc='upper right')
    ax_acc.grid(True, alpha=0.3)
    marker_acc_y, = ax_acc.plot([time[0]], [body_y[0]], 'bo', markersize=8)
    marker_acc_z, = ax_acc.plot(
        [time[0]], [global_z[0]], 'o', color='orange', markersize=8)

    plt.subplots_adjust(bottom=0.12, hspace=0.3, wspace=0.25)

    ax_slider = plt.axes([0.15, 0.03, 0.7, 0.025])
    slider = Slider(ax_slider, 'Time', 0, len(time)-1, valinit=0, valstep=1)

    w, x, y, z = quats[0]
    R = quat_to_matrix(w, x, y, z)
    draw_rocket(ax3d, R)
    ax3d.set_title(f't = {time[0]:.2f}s')

    def update(val):
        idx = int(slider.val)
        w, x, y, z = quats[idx]
        R = quat_to_matrix(w, x, y, z)
        draw_rocket(ax3d, R)
        ax3d.set_title(f't = {time[idx]:.2f}s\nAlt = {baro_alt[idx]:.0f}m')

        marker_baro.set_data([time[idx]], [baro_alt[idx]])
        marker_inert.set_data([time[idx]], [inertial_alt[idx]])
        marker_acc_y.set_data([time[idx]], [body_y[idx]])
        marker_acc_z.set_data([time[idx]], [global_z[idx]])

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == '__main__':
    main()
