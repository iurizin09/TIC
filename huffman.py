from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import komm

with open("alice.txt", "rb") as f:
    entrada = list(f.read())

c = Counter(entrada)
pmf = [c[i] / len(entrada) for i in range(256)]
code = komm.HuffmanCode(pmf)

print([len(x) for x in code.codewords])

#plt.stem(pmf)
#plt.show()
print(f"Entropia da fonte: {komm.entropy(pmf):.2f}")
print(f"Taxa do código: {code.rate(pmf):.2f}")

### Compressão: alice.txt --> alice.huff

codificado = code.encode(entrada)
size = len(codificado)
# pad para multiplo de 8
codificado = np.pad(codificado, (0, 8 - size % 8))

with open("alice.huff", "wb") as f:
    bytes = komm.bits_to_int(codificado.reshape(-1, 8))
    bytes = bytes.astype(np.uint8)
    f.write(bytes)

print(f"Tamanho original   (bits): {len(entrada) * 8}")
print(f"Tamanho comprimido (bits): {len(codificado)}")

### Descompressão: alice.huff --> alice2.txt

with open("alice.huff", "rb") as f:
    bytes = list(f.read())

bits = komm.int_to_bits(bytes, width=8).reshape(-1)
bits = bits[:size]  # remove o pad
saida = code.decode(bits)

with open("alice2.txt", "wb") as f:
    bytes = saida.astype(np.uint8)
    f.write(bytes)