from collections import Counter

import numpy as np
import komm

with open("alice.txt", "rb") as f:
    entrada = list(f.read())

Lz78 = komm.LempelZiv78Code(source_cardinality=256)

### Compressão: alice.txt --> alice.huff

codificado = Lz78.encode(entrada)
size = len(codificado)
# pad para multiplo de 8
codificado = np.pad(codificado, (0, 8 - size % 8))

with open("alice.lz78", "wb") as f:
    bytes = komm.bits_to_int(codificado.reshape(-1, 8))
    bytes = bytes.astype(np.uint8)
    f.write(bytes)

print(f"Tamanho original   (bytes): {len(entrada)} bytes")
print(f"Tamanho comprimido (bytes): {len(codificado)/8}bytes")

### Descompressão: alice.lz77 --> alice2.txt

with open("alice.lz78", "rb") as f:
    bytes = list(f.read())

bits = komm.int_to_bits(bytes, width=8).reshape(-1)
bits = bits[:size]  # remove o pad
saida = Lz78.decode(bits)

with open("alice2.txt", "wb") as f:
    bytes = saida.astype(np.uint8)
    f.write(bytes)