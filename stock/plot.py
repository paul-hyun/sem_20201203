# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


def draw_history(history, ylim=(0, 0.5)):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 1, 1)
    plt.plot(history.history['loss'], 'b-', label='loss')
    plt.plot(history.history['val_loss'], 'r--', label='val_loss')
    plt.xlabel('Epoch')
    plt.ylim(ylim)
    plt.legend()

    plt.show()
