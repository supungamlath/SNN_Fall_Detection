# Open Neuromorphic Seminar - Hands-On with snnTorch
# By Jason K. Eshraghian (www.ncg.ucsc.edu)

from snntorch import spikeplot as splt
import matplotlib.pyplot as plt


def plot_cur_mem_spk(cur, mem, spk, thr_line=False, vline=False, title=False, ylim_max1=1.25, ylim_max2=1.25):
    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.4]})

    # Plot input current
    if cur is not None:
        ax[0].plot(cur, c="tab:orange")
        ax[0].set_ylim([0, ylim_max1])
        ax[0].set_xlim([0, 200])
        ax[0].set_ylabel("Input Current (I_in)")
        if title:
            ax[0].set_title(title)

    # Plot membrane potential
    if mem is not None:
        ax[1].plot(mem)
        ax[1].set_ylim([0, ylim_max2])
        ax[1].set_ylabel("Membrane Potential (U_mem)")
        if thr_line:
            ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
        plt.xlabel("Time step")

    # Plot output spike using spikeplot
    if spk is not None:
        splt.raster(spk, ax[2], s=400, c="black", marker="|")
        if vline:
            ax[2].axvline(
                x=vline,
                ymin=0,
                ymax=6.75,
                alpha=0.15,
                linestyle="dashed",
                c="black",
                linewidth=2,
                zorder=0,
                clip_on=False,
            )
        plt.ylabel("Output spikes")
        plt.yticks([])

    plt.show()


def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, num_steps, title):
    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.4]})

    # Plot input spikes
    splt.raster(spk_in[:, 0], ax[0], s=0.03, c="black")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title(title)

    # Plot hidden layer spikes
    splt.raster(spk1_rec.reshape(num_steps, -1), ax[1], s=0.05, c="black")
    ax[1].set_ylabel("Hidden Layer")

    # Plot output spikes
    splt.raster(spk2_rec.reshape(num_steps, -1), ax[2], c="black", marker="|")
    ax[2].set_ylabel("Output Spikes")
    ax[2].set_ylim([0, 10])

    plt.show()


def dvs_animator(spike_data):
    fig, ax = plt.subplots()
    anim = splt.animator((spike_data[:, 0] + spike_data[:, 1]), fig, ax)
    return anim
