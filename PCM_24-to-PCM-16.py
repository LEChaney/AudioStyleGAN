import soundfile as sf

data, samplerate = sf.read('Lethal_Energies/DS_Heavy_01_Shot-1.wav')
sf.write('Lethal_Energies/DS_Heavy_01_Shot-1_PCM_16.wav', data, samplerate, subtype='PCM_16')

print('Sample Rate: {}'.format(samplerate))