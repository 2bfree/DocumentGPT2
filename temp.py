import numpy as np
import matplotlib.pyplot as plt

# Define the time vector
t = np.linspace(0, 1, 1000)
# Define the signal (a decaying sine wave for example)
signal = np.sin(2 * np.pi * 5 * t) * np.exp(-3 * t)

# Compute the Fourier transform
freqs = np.fft.fftfreq(len(signal), d=(t[1] - t[0]))
fft_vals = np.fft.fft(signal)

# Shift the zero frequency component to the center
fft_vals = np.fft.fftshift(fft_vals)
freqs = np.fft.fftshift(freqs)

# Plot the Fourier transform result
fig, ax = plt.subplots(figsize=(12, 4), facecolor='black')

# Plot the magnitude of the Fourier transform
ax.plot(freqs, np.abs(fft_vals), color='red', linewidth=2)

# Highlight a specific point
highlight_idx = np.argmax(np.abs(fft_vals))
ax.plot(freqs[highlight_idx], np.abs(fft_vals)[highlight_idx], 'wo', markersize=10)

# Customize axis
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(axis='x', colors='gray')
ax.tick_params(axis='y', colors='gray')

# Set limits
ax.set_xlim(-10, 10)
ax.set_ylim(0, np.max(np.abs(fft_vals)) * 1.1)

# Add custom axis arrows
arrowprops = dict(facecolor='gray', edgecolor='gray', arrowstyle='->')
ax.annotate('', xy=(10, 0), xytext=(-10, 0), arrowprops=arrowprops)
ax.annotate('', xy=(0, np.max(np.abs(fft_vals)) * 1.1), xytext=(0, 0), arrowprops=arrowprops)

# Add grid
ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

# Turn off axis labels
ax.set_xticklabels([])
ax.set_yticklabels([])

# Show the plot
plt.tight_layout()
plt.show()
