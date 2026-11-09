import soundfile as sf

info = sf.info("import/brahms-symphony-no-3.flac")
print(info.subtype)          # например 'PCM_16' или 'PCM_24'
print(info.subtype_info)     # текстовое описание, например 'Signed 16 bit PCM'
print(info.format, info.samplerate, info.channels, info.duration)