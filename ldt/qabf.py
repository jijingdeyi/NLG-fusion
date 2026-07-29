import math

import numpy as np
from scipy.signal import convolve2d


def get_Qabf(pA, pB, pF):
    Tg, kg, Dg = 0.9994, -15, 0.5
    Ta, ka, Da = 0.9879, -22, 0.8

    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

    def flip180(arr):
        return np.flip(arr)

    def convolution(k, data):
        k = flip180(k)
        data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        return convolve2d(data, k, mode='valid')

    def get_array(img):
        sax = convolution(h3, img)
        say = convolution(h1, img)
        g = np.sqrt(np.multiply(sax, sax) + np.multiply(say, say))
        a = np.zeros(img.shape)
        zero_mask = sax == 0
        a[~zero_mask] = np.arctan(say[~zero_mask] / sax[~zero_mask])
        a[zero_mask] = np.pi / 2
        return g, a

    gA, aA = get_array(pA)
    gB, aB = get_array(pB)
    gF, aF = get_array(pF)

    def get_qabf(a_src, g_src, a_f, g_f):
        mask = g_src > g_f
        gaf = np.where(
            mask,
            np.divide(g_f, g_src, out=np.zeros_like(g_f), where=g_src != 0),
            np.where(g_src == g_f, g_f, np.divide(g_src, g_f, out=np.zeros_like(g_src), where=g_f != 0)),
        )
        aaf = 1 - np.abs(a_src - a_f) / (math.pi / 2)
        qgaf = Tg / (1 + np.exp(kg * (gaf - Dg)))
        qaaf = Ta / (1 + np.exp(ka * (aaf - Da)))
        return qgaf * qaaf

    qaf = get_qabf(aA, gA, aF, gF)
    qbf = get_qabf(aB, gB, aF, gF)
    deno = np.sum(gA + gB)
    if deno == 0 or not np.isfinite(deno):
        return 0.0
    nume = np.sum(np.multiply(qaf, gA) + np.multiply(qbf, gB))
    out = float(nume / deno)
    return out if np.isfinite(out) else 0.0
