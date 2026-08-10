from collections import Counter
import numpy as np
import komm
import matplotlib.pyplot as plt


with open("alice.txt","rb") as f:

 entrada = list(f.read())

c = Counter(entrada)

pmf = [c[i]/len(entrada) for i in range(256)]

#plt.stem(pmf)
#plt.show()
print(f"Entropia : {komm.entropy(pmf):.2f}")


code = komm.HuffmanCode(pmf)

print(f"Taxa de Codigo : {code.rate(pmf):.2f}")

codificado = code.encode(entrada)

## pad para multiplo de 8
codificado = np.concatenate([codificado,[0,0,0]])

with open("alice.huff", "wb") as f:
 bytes = komm.bits_to_int(codificado.reshape(-1,8))
 bytes = bytes.astype(np.uint8)
 f.write(bytes)

 #struct.pack(f"{len(codificado)}B"),codificado

print(f"Taxa de Codigo (bits): {len(entrada)*8}")

print(f"Tamanho Comprimido (bits): {len(codificado)}")