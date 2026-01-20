import csv
import matplotlib.pyplot as plt
import numpy as np


def local_to_global_ac(accel_local, quat):
    r, i, j, k = quat
    ax, ay, az = accel_local
    rotation_matrix = np.array([
        [1 - 2*(j**2 + k**2), 2*(i*j - r*k), 2*(i*k + r*j)],
        [2*(i*j + r*k), 1 - 2*(i**2 + k**2), 2*(j*k - r*i)],
        [2*(i*k - r*j), 2*(j*k + r*i), 1 - 2*(i**2 + j**2)]
    ])
    accel_vec_local = np.array([ax, ay, az])
    accel_global_standard = np.dot(rotation_matrix, accel_vec_local)
    return accel_global_standard


def integrate(time_data, data: list[float]):
    velocity = [0.0]
    for i in range(1, len(time_data)):
        dt = time_data[i] - time_data[i - 1]
        avg_accel = (data[i] + data[i - 1]) / 2.0
        new_vel = velocity[-1] + (avg_accel * dt)
        velocity.append(new_vel)
    return velocity


def smooth_data(data, window_size=21):
    data = np.array(data)
    kernel = np.ones(window_size) / window_size
    padded = np.pad(data, (window_size//2, window_size//2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed


def derivative(time_data, data):
    time_data = np.array(time_data)
    data = np.array(data)
    dt = np.diff(time_data)
    ddata = np.diff(data)
    velocity = ddata / dt
    velocity = np.insert(velocity, 0, velocity[0])
    return velocity


def plot_inertial_vel_vs_barom(time_data, accel_data, quat_data, altitude_data):
    fig, axes = plt.subplots(5, 1, figsize=(12, 14))
    fig.suptitle('Inertial vs Barometric Comparison',
                 fontsize=14, fontweight='bold')
    global_ac = []
    for idx, acc in enumerate(accel_data):
        i, j, k, real = quat_data[idx]
        # Using body Y acceleration directly (thrust axis)
        global_ac.append(acc[1])
    global_ac = np.array(global_ac)

    global_ac_z = global_ac
    # z is the component we care about
    # take into consideration existing 1g
    vertical_acc = (global_ac_z - 1.0) * 9.80665  # gs to m/s^2
    time_data = np.array(time_data) / 1000.0  # ms to s
    vertical_velocity = integrate(time_data, vertical_acc)  # m/s
    vertical_position = integrate(time_data, vertical_velocity)  # m

    # smooth (altitude data from barom has like 2 decimal places of precision)
    altitude_smoothed = smooth_data(altitude_data, window_size=51)
    barometric_velocity = derivative(time_data, altitude_smoothed)
    barometric_velocity = smooth_data(barometric_velocity, window_size=31)

    # Plot 1: Acceleration
    axes[0].plot(time_data, global_ac_z, 'g-', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Body Y g force (g)', fontsize=10)
    axes[0].set_title('Body Y Acceleration vs Time', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_data, vertical_velocity, 'b-', linewidth=0.5, alpha=0.7,
                 label='Inertial')
    axes[1].plot(time_data, barometric_velocity, 'orange', linewidth=1.0, alpha=0.8,
                 label='Barometric (smoothed)')
    axes[1].set_ylabel('Velocity (m/s)', fontsize=10)
    axes[1].set_title(
        'Velocity Comparison: Inertial vs Barometric', fontsize=11)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_data, vertical_position, 'r-', linewidth=0.5, alpha=0.7)
    axes[2].set_ylabel('Inertial altitude (m)', fontsize=10)
    axes[2].set_title('Inertial Altitude vs Time', fontsize=11)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time_data, altitude_data, 'g-', linewidth=0.5, alpha=0.3,
                 label='Raw')
    axes[3].plot(time_data, altitude_smoothed, 'g-', linewidth=1.5, alpha=0.9,
                 label='Smoothed')
    axes[3].set_ylabel('Barometric Altitude (m)', fontsize=10)
    axes[3].set_title(
        'Barometric Altitude vs Time (Raw + Smoothed)', fontsize=11)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(time_data, barometric_velocity,
                 'orange', linewidth=1.0, alpha=0.8)
    axes[4].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[4].set_ylabel('Barometric Velocity (m/s)', fontsize=10)
    axes[4].set_xlabel('Time (seconds)', fontsize=10)
    axes[4].set_title(
        'Barometric Velocity (from smoothed altitude derivative)', fontsize=11)
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('inertial_vs_barom.png', dpi=300, bbox_inches='tight')
    print(f"Graph saved as 'inertial_vs_barom.png'")
    print(f"Total data points: {len(time_data)}")
    print(f"Time range: {time_data[0]:.2f} to {time_data[-1]:.2f} seconds")


if __name__ == '__main__':
    min_val = 5.875e5
    min_val = 5.75e5
    # max_val = 6.067e5
    max_val = 6.1e5
    min_index = None
    max_index = None

    # Parse file
    time_data = []
    accel_data = []
    quat_data = []
    pressure_data = []
    altitude_data = []

    with open('LOG028.TXT', mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        reader.fieldnames = [h.strip() for h in reader.fieldnames]

        for row in reader:
            values = row['Time'].split(',')
            time, xg, yg, zg = map(float, values[:4])
            pressure, _, altitude = map(float, values[7:10])
            i, j, k, real = map(float, values[13:17])

            if time < min_val or time > max_val:
                continue
            time_data.append(time)
            accel_data.append((xg, yg, zg))
            quat_data.append((i, j, k, real))
            pressure_data.append(pressure)
            altitude_data.append(altitude)

        accel_data = np.array(accel_data)

    plot_inertial_vel_vs_barom(time_data, accel_data, quat_data, altitude_data)
