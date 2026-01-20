import csv
import matplotlib.pyplot as plt
import numpy as np
from inertial_vs_barom import local_to_global_ac

time_data = []
xg_data = []
yg_data = []
zg_data = []
quat_data = []
bno_x_data = []
bno_y_data = []
bno_z_data = []
airbrake_pct_data = []
pressure_data = []

with open('LOG028.TXT', mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter='\t')
    reader.fieldnames = [h.strip() for h in reader.fieldnames]

    for row in reader:
        values = row['Time'].split(',')
        time_data.append(float(values[0]))
        xg_data.append(float(values[1]))
        yg_data.append(float(values[2]))
        zg_data.append(float(values[3]))
        bno_x, bno_y, bno_z = map(float, values[10:13])
        bno_x_data.append(bno_x)
        bno_y_data.append(bno_y)
        bno_z_data.append(bno_z)
        i, j, k, real = map(float, values[13:17])
        quat_data.append((real, i, j, k))
        airbrake_pct_data.append(float(values[18]))
        pressure_data.append(float(values[4]))


def plot_acc(time, xg, yg, zg, airbrake_pct):
    fig, axes = plt.subplots(4, 1, figsize=(16, 10))
    fig.suptitle('Acceleration Components vs Time',
                 fontsize=14, fontweight='bold')

    # Plot Xg vs Time
    axes[0].plot(time, xg, 'r-', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Xg (g)', fontsize=10)
    axes[0].set_title('Xg Acceleration vs Time', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot Yg vs Time
    axes[1].plot(time, yg, 'g-', linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel('Yg (g)', fontsize=10)
    axes[1].set_title('Yg Acceleration vs Time', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Plot Zg vs Time
    axes[2].plot(time, zg, 'b-', linewidth=0.5, alpha=0.7)
    axes[2].set_ylabel('Zg (g)', fontsize=10)
    axes[2].set_xlabel('Time (seconds)', fontsize=10)
    axes[2].set_title('Zg Acceleration vs Time', fontsize=11)
    axes[2].grid(True, alpha=0.3)

    # Plot Airbrake pct vs Time
    axes[3].plot(time, airbrake_pct, 'k-', linewidth=0.5, alpha=0.7)
    axes[3].set_ylabel('Airbrake deployment pct (%)', fontsize=10)
    axes[3].set_xlabel('Time (seconds)', fontsize=10)
    axes[3].set_title('Airbrake pct vs Time', fontsize=11)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('acceleration.png', dpi=300, bbox_inches='tight')
    print(f"Graph saved as 'acceleration.png'")
    print(f"Total data points: {len(time)}")
    print(f"Time range: {time[0]:.2f} to {time[-1]:.2f} seconds")


def plot_acc_vs_ab_pct(time, yg, airbrake_pct):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Acceleration Components vs Time',
                 fontsize=14, fontweight='bold')

    # Plot Yg vs Time
    axes[0].plot(time, yg, 'g-', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Yg (g)', fontsize=10)
    axes[0].set_title('Yg Acceleration vs Time', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot Airbrake pct vs Time
    axes[1].plot(time, airbrake_pct, 'k-', linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel('Airbrake deployment pct (%)', fontsize=10)
    axes[1].set_xlabel('Time (seconds)', fontsize=10)
    axes[1].set_title('Airbrake pct vs Time', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('acc_vs_airbrake.png', dpi=300, bbox_inches='tight')
    print(f"Graph saved as 'acceleration.png'")
    print(f"Total data points: {len(time)}")
    print(f"Time range: {time[0]:.2f} to {time[-1]:.2f} seconds")


def plot_pressure(time, pressure):
    plt.figure()
    plt.plot(time, pressure, 'k-', linewidth=0.5, alpha=0.7)
    plt.xlabel('Time (ms)')
    plt.ylabel('Pressure (hPa)')
    plt.title('Pressure vs Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pressure.png', dpi=300, bbox_inches='tight')


def plot_quaternions(time, quat_list):
    real_vals = [q[0] for q in quat_list]
    i_vals = [q[1] for q in quat_list]
    j_vals = [q[2] for q in quat_list]
    k_vals = [q[3] for q in quat_list]

    magnitudes = [np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
                  for q in quat_list]

    quat_arr = np.array(quat_list)
    quat_diff = np.linalg.norm(np.diff(quat_arr, axis=0), axis=1)
    quat_diff = np.insert(quat_diff, 0, 0)

    fig, axes = plt.subplots(6, 1, figsize=(16, 14))
    fig.suptitle('Raw Quaternion Components vs Time',
                 fontsize=14, fontweight='bold')

    axes[0].plot(time, real_vals, 'purple', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Real (w)', fontsize=10)
    axes[0].set_title('Quaternion Real Component', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-1.1, 1.1)

    axes[1].plot(time, i_vals, 'r-', linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel('i', fontsize=10)
    axes[1].set_title('Quaternion i Component', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-1.1, 1.1)

    axes[2].plot(time, j_vals, 'g-', linewidth=0.5, alpha=0.7)
    axes[2].set_ylabel('j', fontsize=10)
    axes[2].set_title('Quaternion j Component', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-1.1, 1.1)

    axes[3].plot(time, k_vals, 'b-', linewidth=0.5, alpha=0.7)
    axes[3].set_ylabel('k', fontsize=10)
    axes[3].set_title('Quaternion k Component', fontsize=11)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim(-1.1, 1.1)

    axes[4].plot(time, magnitudes, 'k-', linewidth=0.5, alpha=0.7)
    axes[4].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[4].set_ylabel('Magnitude', fontsize=10)
    axes[4].set_title('Quaternion Magnitude (should be 1.0)', fontsize=11)
    axes[4].grid(True, alpha=0.3)
    axes[4].set_ylim(0.9, 1.1)

    axes[5].plot(time, quat_diff, 'orange', linewidth=0.5, alpha=0.7)
    axes[5].set_ylabel('Δ Quaternion', fontsize=10)
    axes[5].set_xlabel('Time (ms)', fontsize=10)
    axes[5].set_title('Quaternion Rate of Change', fontsize=11)
    axes[5].grid(True, alpha=0.3)
    axes[5].legend()

    plt.tight_layout()
    plt.savefig('quaternion_raw.png', dpi=300, bbox_inches='tight')
    print(f"Graph saved as 'quaternion_raw.png'")
    print(f"Total data points: {len(time)}")
    print(f"Time range: {time[0]:.2f} to {time[-1]:.2f} ms")


def plot_euler_angles(time, bno_x, bno_y, bno_z):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle('BNO055 Euler Angles vs Time',
                 fontsize=14, fontweight='bold')

    axes[0].plot(time, bno_x, 'r-', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('BNO_X (degrees)', fontsize=10)
    axes[0].set_title('BNO_X (Heading/Yaw)', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, bno_y, 'g-', linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel('BNO_Y (degrees)', fontsize=10)
    axes[1].set_title('BNO_Y (Roll)', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, bno_z, 'b-', linewidth=0.5, alpha=0.7)
    axes[2].set_ylabel('BNO_Z (degrees)', fontsize=10)
    axes[2].set_xlabel('Time (ms)', fontsize=10)
    axes[2].set_title('BNO_Z (Pitch)', fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('euler_angles.png', dpi=300, bbox_inches='tight')
    print(f"Graph saved as 'euler_angles.png'")
    print(f"Total data points: {len(time)}")
    print(f"Time range: {time[0]:.2f} to {time[-1]:.2f} ms")


def plot_body_vs_global(time, xg, yg, zg, quat_list):
    global_x, global_y, global_z = [], [], []
    for i in range(len(time)):
        acc_body = (xg[i], yg[i], zg[i])
        quat = quat_list[i]
        acc_global = local_to_global_ac(acc_body, quat)
        global_x.append(acc_global[0])
        global_y.append(acc_global[1])
        global_z.append(acc_global[2])

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Body Frame vs Global Frame Accelerations',
                 fontsize=14, fontweight='bold')

    axes[0, 0].plot(time, xg, 'r-', linewidth=0.5, alpha=0.7)
    axes[0, 0].set_ylabel('Xg (g)', fontsize=10)
    axes[0, 0].set_title('Body Frame: Xg vs Time', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(time, yg, 'g-', linewidth=0.5, alpha=0.7)
    axes[1, 0].set_ylabel('Yg (g)', fontsize=10)
    axes[1, 0].set_title('Body Frame: Yg vs Time', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    axes[2, 0].plot(time, zg, 'b-', linewidth=0.5, alpha=0.7)
    axes[2, 0].set_ylabel('Zg (g)', fontsize=10)
    axes[2, 0].set_xlabel('Time (ms)', fontsize=10)
    axes[2, 0].set_title('Body Frame: Zg vs Time', fontsize=11)
    axes[2, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(time, global_x, 'r-', linewidth=0.5, alpha=0.7)
    axes[0, 1].set_ylabel('Global X (g)', fontsize=10)
    axes[0, 1].set_title('Global Frame: X vs Time', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].plot(time, global_y, 'g-', linewidth=0.5, alpha=0.7)
    axes[1, 1].set_ylabel('Global Y (g)', fontsize=10)
    axes[1, 1].set_title('Global Frame: Y vs Time', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 1].plot(time, global_z, 'b-', linewidth=0.5, alpha=0.7)
    axes[2, 1].set_ylabel('Global Z (g) [Vertical]', fontsize=10)
    axes[2, 1].set_xlabel('Time (ms)', fontsize=10)
    axes[2, 1].set_title('Global Frame: Z (Vertical) vs Time', fontsize=11)
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 1].axhline(y=1.0, color='gray', linestyle='--',
                       alpha=0.5, label='1g baseline')
    axes[2, 1].legend()

    plt.tight_layout()
    plt.savefig('body_vs_global_acc.png', dpi=300, bbox_inches='tight')
    print(f"Graph saved as 'body_vs_global_acc.png'")
    print(f"Total data points: {len(time)}")
    print(f"Time range: {time[0]:.2f} to {time[-1]:.2f} ms")


if __name__ == '__main__':
    # min_val = 5.75e5
    # max_val = 6.25e5

    min_val = 5.75e5
    max_val = 7e5

    min_index = None
    max_index = None
    for index, val in enumerate(time_data):
        if val > min_val and min_index is None:
            min_index = index
        if val > max_val and max_index is None:
            max_index = index

    print(min_index, max_index)
    time_data = time_data[min_index:max_index]
    xg_data = xg_data[min_index:max_index]
    yg_data = yg_data[min_index:max_index]
    zg_data = zg_data[min_index:max_index]
    quat_data = quat_data[min_index:max_index]
    bno_x_data = bno_x_data[min_index:max_index]
    bno_y_data = bno_y_data[min_index:max_index]
    bno_z_data = bno_z_data[min_index:max_index]
    airbrake_pct_data = airbrake_pct_data[min_index:max_index]
    pressure_data = pressure_data[min_index:max_index]

    plot_euler_angles(time_data, bno_x_data, bno_y_data, bno_z_data)
    plot_quaternions(time_data, quat_data)
    plot_body_vs_global(time_data, xg_data, yg_data, zg_data, quat_data)
    plot_acc_vs_ab_pct(time_data, yg_data, airbrake_pct_data)
    plot_acc(time_data, xg_data, yg_data, zg_data, airbrake_pct_data)
    plot_pressure(time_data, pressure_data)
