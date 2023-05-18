import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_matrix(mat, out_path):
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(mat, interpolation='nearest', cmap='rainbow')
    fig.colorbar(cax)
    fig.savefig(out_path)
    print(f'Save fig on {out_path}')


def plot_scatter_single(iou1, iou2, err, out_path):
    idx = np.argsort(err)
    iou1 = iou1[idx]
    iou2 = iou2[idx]
    err = err[idx]
    
    plt.rcParams['axes.unicode_minus'] = False 

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel("IoU", fontsize=13, fontweight='bold')
    plt.ylabel("angle", fontsize=13, fontweight='bold')

    cm=plt.cm.get_cmap('rainbow')
    sc=plt.scatter(iou2, iou1,c=err,cmap=cm)
    plt.colorbar(sc)

    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(out_path)
    print(f'Save fig on {out_path}')


def plot_curve(data, out_path):
    plt.rcParams['axes.unicode_minus'] = False

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for args in data.values():
        plt.plot(args['x'], args['y'], label=args['label'])
    plt.legend(loc='upper left')

    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_path)
    print(f'Save fig on {out_path}')


def plot_scatter(data, out_path, all_in_one=False, show_text=True, grid=(1,3)):
    if all_in_one:
        ax = plt.gca()
        ax.axis([0, 1, 0, 1])
        ax.grid()
        ax.set_aspect(1)
        for args in data:
            ax.scatter(**args, s=2, alpha=0.5)
        if show_text:
            ax.legend(loc='upper left')
    else:
        fig, axes = plt.subplots(*grid, figsize=(3*grid[1]+0.01, 3*grid[0]+0.01), layout="constrained")
        for idx, (ax, args) in enumerate(zip(axes.flatten(), data)):
            ax.grid()
            ax.set_aspect(1)
            ax.axis([0, 1, 0, 1])

            xy_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            xy_labels = ['0.0', '', '', '', '', '1.0']
            ax.set(xticks=xy_ticks, xticklabels=xy_labels)
            ax.set(yticks=xy_ticks, yticklabels=xy_labels)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)

            R = args.pop('R')['R']
            #print(R)
            ax.text(0.5, 0.1, f'R={R:.4f}', fontsize=15, va='center', fontweight='bold', color=args['color'])
            ax.scatter(**args, s=2)
            if show_text and idx < 3:
                ax.set_title(args['label'], fontsize=15, fontweight='bold', pad=10)
        if show_text:
            fig.text(0.51, 0.02, 'Approximate IoU', fontsize=18, va='center', ha='center',  fontweight='bold')
            fig.text(0.02, 0.5, 'Unbiased IoU', fontsize=18, va='center', ha='center', rotation='vertical', fontweight='bold')

        plt.tight_layout()
        fig.subplots_adjust(left=0.08, bottom=0.065)

    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(out_path)
    print(f'Save fig on {out_path}')
