# Generator Spektrograma - Alat za Analizu Audio Datoteka

Ovaj Python alat omogućava automatsko generiranje različitih vrsta spektrograma i vizualizacija iz audio datoteka. Idealan je za glazbene producente, audio inženjere, istraživače i sve koji žele analizirati karakteristike zvuka.

## Značajke

### Vrste Spektrograma
- **Standardni spektrogram** - Prikazuje frekvenciju preko vremena
- **Mel spektrogram** - Optimiziran za ljudsku percepciju zvuka
- **Chromagram** - Analizira melodijski sadržaj i harmonije
- **Harmonijski spektrogram** - Izdvaja harmonijske komponente
- **Perkusijski spektrogram** - Izdvaja ritmičke elemente

### Napredne Vizualizacije
- **Kombinirana analiza** - Sve analize u jednoj slici (waveform, spektrogram, mel spektrogram, chromagram)
- **Analiza značajki** - Usporedba spektralne centroide i tempa između datoteka
- **Scatter plot** - Korelacija između audio karakteristika

### Dodatne Mogućnosti
- Paralelno procesiranje više datoteka
- Podrška za različite audio formate (.wav, .mp3, .ogg, .flac)
- Prilagodljivi parametri analize
- Automatsko stvaranje direktorija za organizaciju rezultata

## Instalacija

### Potrebne Biblioteke
```bash
pip install librosa numpy matplotlib tqdm
```

### Sistemski Zahtjevi
- Python 3.7+
- Preporučeno: 8GB RAM za veće datoteke
- FFmpeg (za .mp3 i druge kompresije formate)

## Korištenje

### Osnovna Komanda
```bash
python spectrogram_generator.py --input_folder /putanja/do/audio/datoteka --audio_type .wav
```

### Sve Dostupne Opcije
```bash
python spectrogram_generator.py \
    --input_folder ./audio \
    --output_folder ./rezultati \
    --audio_type .wav \
    --n_fft 2048 \
    --hop_length 512 \
    --n_mels 128 \
    --fmin 20 \
    --fmax 8000 \
    --sample_rate 22050
```

### Parametri

| Parametar | Opis | Zadana Vrijednost |
|-----------|------|-------------------|
| `--input_folder` | Direktorij s audio datotekama | **Obavezno** |
| `--output_folder` | Direktorij za spremanje rezultata | `spectograms` |
| `--audio_type` | Tip datoteke (.wav, .mp3, .ogg, .flac) | **Obavezno** |
| `--n_fft` | Veličina FFT prozora | `2048` |
| `--hop_length` | Broj uzoraka između okvira | `512` |
| `--n_mels` | Broj mel bendova | `128` |
| `--fmin` | Najniža frekvencija (Hz) | `20` |
| `--fmax` | Najviša frekvencija (Hz) | `8000` |
| `--sample_rate` | Ciljna frekvencija uzorkovanja | `None` (izvorna) |

## Struktura Izlaznih Direktorija

```
rezultati/
├── spectrograms/           # Standardni spektrogrami
├── mel_spectrograms/       # Mel spektrogrami  
├── chroma/                 # Chromagrami
├── harmonic/              # Harmonijski spektrogrami
├── percussive/            # Perkusijski spektrogrami
├── advanced_visualizations/ # Kombinirana analiza
└── summary/               # Usporedne analize
    ├── spectral_centroids.png
    ├── tempos.png
    └── centroid_vs_tempo.png
```

## Primjeri Korištenja

### Analiza Glazbene Kolekcije
```bash
# Analiziraj sve WAV datoteke u direktoriju "moja_glazba"
python spectrogram_generator.py \
    --input_folder ./moja_glazba \
    --output_folder ./analiza_glazbe \
    --audio_type .wav
```

### Detaljana Analiza s Prilagođenim Parametrima
```bash
# Visoka rezolucija za detaljnu analizu
python spectrogram_generator.py \
    --input_folder ./zvukovi \
    --output_folder ./detaljno \
    --audio_type .flac \
    --n_fft 4096 \
    --hop_length 256 \
    --n_mels 256 \
    --sample_rate 44100
```

### Analiza Govornih Zapisa
```bash
# Optimizirano za govor (niži frekvencijski opseg)
python spectrogram_generator.py \
    --input_folder ./govori \
    --audio_type .mp3 \
    --fmax 4000 \
    --n_mels 64
```

## Objašnjenje Rezultata

### Spektrogram
Pokazuje kako se frekvencijski sadržaj mijenja kroz vrijeme. Svjetlije boje označavaju jače frekvencije.

### Mel Spektrogram  
Prilagođen ljudskom sluhu - korisno za prepoznavanje govora i glazbenu analizu.

### Chromagram
Prikazuje distribuciju energy po notama (C, C#, D, D#, itd.) - odličan za analizu harmonije.

### Harmonijski/Perkusijski Spektrogram
Razdvaja melodijske elemente od ritmičkih - korisno za remixiranje i produkciju.

### Analiza Značajki
- **Spektralna centroida** - "svjetlina" zvuka (više = svjetliji zvuk)
- **Tempo** - procjena brzine u BPM
- **Scatter plot** - pokazuje odnos između svjetline i tempa

## Savjeti za Optimizaciju

### Performanse
- Koristite SSD za brže učitavanje datoteka
- Smanjite `n_fft` i `n_mels` za brže procesiranje
- Povećajte broj worker procesa za moćnije računala

### Kvaliteta Analize
- Povećajte `n_fft` za bolju frekvencijsku rezoluciju
- Smanjite `hop_length` za bolju vremensku rezoluciju
- Prilagodite `fmin` i `fmax` prema sadržaju (govor vs. glazba)

## Rješavanje Problema

### Greške pri Učitavanju
```
Error: NoBackendError
```
**Rješenje:** Instalirajte FFmpeg ili koristite .wav datoteke

### Nedostatak Memorije
```
MemoryError
```
**Rješenje:** Smanjite `n_fft`, `n_mels` ili procesiranje po manjim skupinama

### Prazni Rezultati  
**Provjera:** Provjerite putanje direktorija i ekstenzije datoteka

## Dodatne Mogućnosti

### Batch Procesiranje
Script automatski pronalazi sve datoteke zadanog tipa u direktoriju i procesira ih paralelno.

### Prilagodba za Različite Slučajeve Korištenja
- **Glazbena produkcija:** Povećajte `n_mels` i koristite punu frekvencijsku skalu
- **Analiza govora:** Ograničite na 4000Hz, smanjite `n_mels`  
- **Zvukovi prirode:** Koristite široki frekvencijski opseg

## Autor i Licenca

Ovaj alat je stvoren za potrebe audio analize i istraživanja. Slobodno ga koristite i prilagođavajte svojim potrebama.

### Potrebna Priznanja
- **librosa** - za audio procesiranje
- **matplotlib** - za vizualizacije  
- **numpy** - za numeričke operacije

---

*Za dodatna pitanja ili prijedloge, otvorite issue na GitHub repozitoriju.*