import whisper
from pyannote.audio import Pipeline
import librosa 
import torch
import os

# Whisper modelini yükle
whisper_model = whisper.load_model("base")

# Pipeline'ı yükle
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token="HFAKQczN4KeuhBpf7XN7wdrIBqL1lSm")

def ses_isleme_fonksiyonu(dosya_yolu: str):
    print(f"İşleniyor: {dosya_yolu}")

    # 1. Ses dosyasını metne çevirme
    result = whisper_model.transcribe(dosya_yolu)
    metin = result["text"]

    # 2. Konuşmacıları ayırma (Pyannote)
    waveform, sample_rate = librosa.load(dosya_yolu, sr=16000)
    waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
    
    # 3. Pipeline'ı çalıştır
    diarization_output = pipeline({"waveform": waveform_tensor, "sample_rate": sample_rate})

    # 4. KABA KUVVET ÇÖZÜMÜ: Veriyi zorla çekme
    konusmalar = []
    gercek_veri = None
    
    # Adım 4a: Çıktı doğrudan itertracks'e sahip mi?
    if hasattr(diarization_output, 'itertracks'):
        gercek_veri = diarization_output
    else:
        # Adım 4b: Sahip değilse, çıktının içindeki TÜM özellikleri tara ve itertracks'i bul!
        for ozellik in dir(diarization_output):
            if not ozellik.startswith("_"):  # Gizli olmayan özelliklere bak
                try:
                    alt_nesne = getattr(diarization_output, ozellik)
                    if hasattr(alt_nesne, 'itertracks'):
                        gercek_veri = alt_nesne
                        break
                except:
                    continue

    # Adım 5: Veriyi listeye dök
    if gercek_veri is not None:
        for turn, _, speaker in gercek_veri.itertracks(yield_label=True):
            konusmalar.append(f"{speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
    else:
        # Adım 6: Eğer hiçbir şey bulamazsak sistemi ÇÖKERTME!
        # Pyannote nesneleri metne çevrildiğinde "[00:00:00 --> 00:00:01] SPEAKER" formatında yazdırılır.
        # Direkt bu string'i alıp bölerek kullanıyoruz.
        text_output = str(diarization_output)
        for satir in text_output.split('\n'):
            if satir.strip():
                konusmalar.append(satir.strip())

    return {"metin": metin, "konusmalar": konusmalar}
    