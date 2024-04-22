import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la simulación OFDM
num_subcarriers = 64        # Número de subportadoras
num_symbols = 1000          # Número de símbolos a transmitir
cp_len = 16                 # Longitud del prefijo cíclico

# Generación de bits aleatorios para que coincidan con el número de símbolos y subportadoras
num_bits = 2 * num_subcarriers * num_symbols  # Aseguramos que el número de bits sea el correcto
bits = np.random.randint(0, 2, num_bits)

# Modulación QPSK
symbols = 1 - 2 * bits[0::2] + 1j * (1 - 2 * bits[1::2])
symbols = symbols.reshape(num_symbols, num_subcarriers)

# IDFT
ofdm_time = np.fft.ifft(symbols, axis=1) * np.sqrt(num_subcarriers)

# Agregar prefijo cíclico
ofdm_with_cp = np.hstack((ofdm_time[:, -cp_len:], ofdm_time))

# Emulación de la transmisión (en este caso, no hay canal ni ruido)
received_signal = ofdm_with_cp

# Remover prefijo cíclico
received_signal = received_signal[:, cp_len:]

# DFT para volver al dominio de la frecuencia
received_freq = np.fft.fft(received_signal, axis=1) / np.sqrt(num_subcarriers)

# Demodulación QPSK
received_bits = np.vstack((received_freq.real > 0, received_freq.imag > 0)).astype(int).T

# Cálculo del BER (Bit Error Rate)
errors = (received_bits.flatten() != bits)
ber = errors.sum() / len(bits)

# Visualización de un símbolo OFDM en el tiempo
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(np.abs(ofdm_time[0]), label='Symbol Magnitude')
plt.title('Magnitud de un Símbolo OFDM en el Tiempo')
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.angle(ofdm_time[0]), label='Symbol Phase')
plt.title('Fase de un Símbolo OFDM en el Tiempo')
plt.xlabel('Samples')
plt.ylabel('Phase (radians)')
plt.legend()

plt.tight_layout()
plt.show()

print(f"La Tasa de Error de Bit (BER) es: {ber:.6f}")
