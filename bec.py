import numpy as np
import komm
import matplotlib.pyplot as plt

eps = 0.2

M_rep3 = 2
M_spc3 = 4
M = M_rep3

rep3 = komm.RepetitionCode(3)
# spc3 = komm.SingleParityCheckCode(3)
labeling = komm.NaturalLabeling(1)
bec = komm.BinaryErasureChannel(eps)
dms = komm.DiscreteMemorylessSource(M)

codebook_rep3 = [[0, 0, 0], [1, 1, 1]]

codebook_spc3 = [[0, 0, 0],[0, 1, 1],[1, 0, 1],[1, 1, 0]]

P_erro_teo_rep3 = eps**3/2
P_erro_teo_spc3 = 3*eps**2 / 2 - 3*eps**3 / 4 

codebook = codebook_rep3


def eh_compativel(y, x):
    for i in range(len(y)):
        if y[i] != 2 and y[i] != x[i]:
            return False
    return True


def decode(y):
    for i in range(len(codebook)):
        x = codebook[i]
        if eh_compativel(y, x):
            return i

    return None


m = dms.emit(1)[0]

eps_values = np.linspace(0, 1, 50)

P_erro_rep3 = []
P_erro_spc3 = []

for eps in eps_values:

    bec = komm.BinaryErasureChannel(eps)

    # REP-3
    codebook = codebook_rep3
    erros = 0

    for i in range(10_000):

        m = np.random.randint(2)

        x = codebook[m]
        y = bec.transmit(x).tolist()

        m_hat = decode(y)

        if m_hat != m:
            erros += 1

    P_erro_rep3.append(erros / 10_000)


    # SPC-3
    codebook = codebook_spc3
    erros = 0

    for i in range(10_000):

        m = np.random.randint(4)

        x = codebook[m]
        y = bec.transmit(x).tolist()

        m_hat = decode(y)

        if m_hat != m:
            erros += 1

    P_erro_spc3.append(erros / 10_000)


# =========================
# PLOT
# =========================

plt.figure(figsize=(8, 5))

# REP-3
plt.plot(
    eps_values,
    P_erro_rep3,
    'o',
    markersize=4,
    label='REP-3 Simulado'
)

plt.plot(
    eps_values,
    eps_values**3 / 2,
    '-',
    label='REP-3 Teórico'
)

# SPC-3
plt.plot(
    eps_values,
    P_erro_spc3,
    'o',
    markersize=4,
    label='SPC-3 Simulado'
)

plt.plot(
    eps_values,
    3 * eps_values**2 / 2 - 3 * eps_values**3 / 4,
    '-',
    label='SPC-3 Teórico'
)

plt.xlabel('ε')
plt.ylabel('Probabilidade de erro')
plt.title('BEC - REP-3 e SPC-3')

plt.grid()
plt.legend()
plt.show()
