import matplotlib.pyplot as plt
import librosa.display
from scipy.signal import butter, filtfilt
import soundfile as sf

# Carregando o arquivo de áudio
audio1 = 'vazado_1.wav'
audio2 = 'vazado_3.wav'

# Carregando áudio com sr=32000
y, sr = librosa.load(audio2, sr=32000)

# Exibindo forma de onda original
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title("Forma de onda original")
plt.tight_layout()

print("Sinal convertido")

# Função do filtro passa-faixa
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs  # Frequência de Nyquist
    low = lowcut / nyquist  # Normalizar a frequência de corte inferior
    high = highcut / nyquist  # Normalizar a frequência de corte superior
    b, a = butter(order, [low, high], btype='band')  # Criar o filtro passa-faixa
    y_filtered = filtfilt(b, a, data)  # Aplicar o filtro
    return y_filtered

# Definindo as frequências de corte
low_cutoff = 500  # Frequência de corte inferior (Hz)
high_cutoff = 1600  # Frequência de corte superior (Hz)

# Aplicando o filtro passa-faixa

filtered_signal = bandpass_filter(y, low_cutoff, high_cutoff, sr)

# Amplificador
filtered_signal = filtered_signal * 20

# Plotando a forma de onda filtrada
plt.subplot(2, 1, 2)
librosa.display.waveshow(filtered_signal, sr=sr)
plt.title("Forma de onda filtrada")
plt.tight_layout()
plt.show()


# Salvando o áudio filtrado em um arquivo
sf.write('audio_filtrado.wav', filtered_signal, sr)

print("Áudio filtrado salvo como 'audio_filtrado.wav'")
