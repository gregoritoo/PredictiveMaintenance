import matplotlib.pyplot as plt
import numpy as np

color_dic = {
    "BROKEN": "black",
    "MAINTENANCE": "green",
    "NORMAL": "blue",
    "DANGER_ZONE": "orange",
    "URGENT_DANGER_ZONE": "red",
}

size_dic = {"BROKEN": 50, "MAINTENANCE": 2, "NORMAL": 1, "DANGER_ZONE": 2, "URGENT_DANGER_ZONE": 2}

alpha_dic = {
    "BROKEN": 1,
    "MAINTENANCE": 0.5,
    "NORMAL": 0.2,
    "DANGER_ZONE": 0.8,
    "URGENT_DANGER_ZONE": 0.8,
}


def plot_pca(imputed_df, projected_data):
    projected_data = np.array(projected_data)
    labels = imputed_df.predictive_machine_status.values

    unique_labels = imputed_df.predictive_machine_status.unique()

    for label in unique_labels[:-1]:

        idx = np.where(labels == label)

        points = projected_data[idx]

        plt.scatter(
            points[:, 0],
            points[:, 1],
            color=color_dic[label],
            label=label,
            s=size_dic[label],
            alpha=alpha_dic[label],
        )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()


def plot_pca_by_cycle(projected_data, imputed_df, ax, cycle=None):
    if cycle is not None:
        idx = list(imputed_df[imputed_df["Cycle"] == cycle].index)
        projected_data = np.array(projected_data)[idx]
        labels = imputed_df.loc[idx, "predictive_machine_status"].values
    else:
        projected_data = np.array(projected_data)
        labels = imputed_df.predictive_machine_status.values

    unique_labels = imputed_df.predictive_machine_status.unique()

    for label in unique_labels[:-1]:

        idx = np.where(labels == label)

        points = projected_data[idx]

        ax.scatter(
            points[:, 0],
            points[:, 1],
            color=color_dic[label],
            label=label,
            s=size_dic[label],
            alpha=alpha_dic[label],
        )
        ax.set_title(f"Cycle {cycle}")
    return ax


def plot_psd(time_serie, fs=1):
    plt.figure()

    if len(time_serie) > 0:
        n = len(time_serie)
        fft_result = np.fft.fft(time_serie)
        frequencies = np.fft.fftfreq(n, d=1 / fs)

        PSD = (np.abs(fft_result) ** 2) / n
        positive_freq_indices = frequencies >= 0

        plt.plot(frequencies[positive_freq_indices], PSD[positive_freq_indices], label="PSD")
        plt.title("Power Spectral Density (PSD)")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power")
        plt.legend()
        plt.grid()
        plt.show()
