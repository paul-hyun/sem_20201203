import os

import matplotlib.pyplot as plt


def draw_history(args, history, ylim=(0, 0.5)):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 1, 1)
    plt.plot(history.history['loss'], 'b-', label='loss')
    plt.plot(history.history['val_loss'], 'r--', label='val_loss')
    plt.xlabel('Epoch')
    # plt.ylim(ylim)
    plt.legend()

    plt.savefig(os.path.join(args.out_dir, "history.png"))
    plt.show()


def draw_pred(args, y_true, y_pred):
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 1, 1)
    plt.plot(y_true, 'b-', label='y_true')
    plt.plot(y_pred, 'r--', label='y_pred')
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, "pred.png"))
    plt.show()


def draw_error(args, y_true, y_pred):
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 1, 1)
    plt.plot(y_true - y_pred, 'r-', label='error')
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, "error.png"))
    plt.show()
