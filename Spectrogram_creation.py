import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


class SpectrogramGenerator:
    def __init__(self, input_folder, output_folder,
                 n_fft=2048, hop_length=512, n_mels=128,
                 fmin=20, fmax=8000, sample_rate=None, file_type = ".wav"):
        """
        Initialize the SpectrogramGenerator class.

        Parameters:
        -----------
        input_folder : str
            Path to folder containing audio files.
        output_folder : str
            Path to folder where spectrograms will be saved.
        n_fft : int
            FFT window size.
        hop_length : int
            Number of samples between successive frames.
        n_mels : int
            Number of mel bands to generate.
        fmin : int
            Lowest frequency (in Hz).
        fmax : int
            Highest frequency (in Hz).
        sample_rate : int or None
            Target sample rate. If None, uses the native sample rate of each file.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.sample_rate = sample_rate
        self.file_type = file_type

        self.spectrogram_dir = os.path.join(output_folder, 'spectrograms')
        self.melspectrogram_dir = os.path.join(output_folder, 'mel_spectrograms')
        self.chroma_dir = os.path.join(output_folder, 'chroma')
        self.harmonic_dir = os.path.join(output_folder, 'harmonic')
        self.percussive_dir = os.path.join(output_folder, 'percussive')

        for directory in [self.spectrogram_dir, self.melspectrogram_dir,
                          self.chroma_dir, self.harmonic_dir, self.percussive_dir]:
            os.makedirs(directory, exist_ok=True)

        self.audio_files = [f for f in os.listdir(input_folder)
                            if f.lower().endswith(self.file_type)]
        print(f"Found {len(self.audio_files)} WAV files.")

    def _load_audio(self, filename):
        """Load an audio file and return the signal."""
        file_path = os.path.join(self.input_folder, filename)
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        return y, sr

    def _save_figure(self, fig, output_path):
        """Save a matplotlib figure to a file."""
        canvas = FigureCanvas(fig)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def generate_standard_spectrogram(self, filename):
        """Generate and save a standard spectrogram."""
        y, sr = self._load_audio(filename)

        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        fig, ax = plt.subplots(figsize=(10, 5))
        img = librosa.display.specshow(D_db, x_axis='time', y_axis='log',
                                       sr=sr, hop_length=self.hop_length, ax=ax)
        ax.set_title(f'Spectrogram - {os.path.basename(filename)}')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')

        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(self.spectrogram_dir, f"{base_name}_spectrogram.png")
        self._save_figure(fig, output_path)

        return D, sr

    def generate_mel_spectrogram(self, filename):
        """Generate and save a mel spectrogram."""
        y, sr = self._load_audio(filename)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft,
                                           hop_length=self.hop_length, n_mels=self.n_mels,
                                           fmin=self.fmin, fmax=self.fmax)
        S_db = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(10, 5))
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel',
                                       sr=sr, hop_length=self.hop_length,
                                       fmin=self.fmin, fmax=self.fmax, ax=ax)
        ax.set_title(f'Mel Spectrogram - {os.path.basename(filename)}')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')

        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(self.melspectrogram_dir, f"{base_name}_melspectrogram.png")
        self._save_figure(fig, output_path)

        return S, sr

    def generate_chromagram(self, filename):
        """Generate and save a chromagram (melodic spectrogram)."""
        y, sr = self._load_audio(filename)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=self.n_fft,
                                             hop_length=self.hop_length)

        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma',
                                       sr=sr, hop_length=self.hop_length, ax=ax)
        ax.set_title(f'Chromagram - {os.path.basename(filename)}')
        fig.colorbar(img, ax=ax)

        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(self.chroma_dir, f"{base_name}_chroma.png")
        self._save_figure(fig, output_path)

        return chroma, sr

    def generate_harmonic_percussive_spectrograms(self, filename):
        """Generate and save harmonic and percussive spectrograms."""
        y, sr = self._load_audio(filename)

        y_harmonic, y_percussive = librosa.effects.hpss(y)

        D_harmonic = librosa.stft(y_harmonic, n_fft=self.n_fft, hop_length=self.hop_length)
        D_percussive = librosa.stft(y_percussive, n_fft=self.n_fft, hop_length=self.hop_length)

        D_harmonic_db = librosa.amplitude_to_db(np.abs(D_harmonic), ref=np.max)
        D_percussive_db = librosa.amplitude_to_db(np.abs(D_percussive), ref=np.max)

        fig_harm, ax_harm = plt.subplots(figsize=(10, 5))
        img_harm = librosa.display.specshow(D_harmonic_db, x_axis='time', y_axis='log',
                                            sr=sr, hop_length=self.hop_length, ax=ax_harm)
        ax_harm.set_title(f'Harmonic Spectrogram - {os.path.basename(filename)}')
        fig_harm.colorbar(img_harm, ax=ax_harm, format='%+2.0f dB')

        base_name = os.path.splitext(filename)[0]
        output_path_harm = os.path.join(self.harmonic_dir, f"{base_name}_harmonic.png")
        self._save_figure(fig_harm, output_path_harm)

        fig_perc, ax_perc = plt.subplots(figsize=(10, 5))
        img_perc = librosa.display.specshow(D_percussive_db, x_axis='time', y_axis='log',
                                            sr=sr, hop_length=self.hop_length, ax=ax_perc)
        ax_perc.set_title(f'Percussive Spectrogram - {os.path.basename(filename)}')
        fig_perc.colorbar(img_perc, ax=ax_perc, format='%+2.0f dB')

        output_path_perc = os.path.join(self.percussive_dir, f"{base_name}_percussive.png")
        self._save_figure(fig_perc, output_path_perc)

        return (D_harmonic, D_percussive), sr

    def process_file(self, filename):
        """Process a single audio file to generate all spectrograms."""
        try:
            print(f"Processing {filename}...")
            self.generate_standard_spectrogram(filename)
            self.generate_mel_spectrogram(filename)
            self.generate_chromagram(filename)
            self.generate_harmonic_percussive_spectrograms(filename)
            return True
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return False

    def process_all_files(self, num_workers=4):
        """Process all audio files in the input folder using multiple processes."""
        successful = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(self.process_file, self.audio_files),
                                total=len(self.audio_files)))

        successful = sum(results)
        failed = len(results) - successful

        print(f"Completed processing {len(self.audio_files)} files.")
        print(f"Successful: {successful}, Failed: {failed}")


def create_advanced_visualization(input_folder, output_folder, filename):
    """
    Create a more advanced visualization for a single audio file with multiple spectrograms.

    Parameters:
    -----------
    input_folder : str
        Path to folder containing audio files.
    output_folder : str
        Path to folder where visualizations will be saved.
    filename : str
        Name of the audio file to process.
    """
    y, sr = librosa.load(os.path.join(input_folder, filename))

    advanced_viz_dir = os.path.join(output_folder, 'advanced_visualizations')
    os.makedirs(advanced_viz_dir, exist_ok=True)

    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    librosa.display.waveshow(y, sr=sr, ax=axs[0])
    axs[0].set_title('Waveform')
    axs[0].set_ylabel('Amplitude')

    D = librosa.stft(y, n_fft=2048, hop_length=512)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img1 = librosa.display.specshow(D_db, x_axis='time', y_axis='log', sr=sr,
                                    hop_length=512, ax=axs[1])
    axs[1].set_title('Spectrogram')
    fig.colorbar(img1, ax=axs[1], format='%+2.0f dB')

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    img2 = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr,
                                    hop_length=512, ax=axs[2])
    axs[2].set_title('Mel Spectrogram')
    fig.colorbar(img2, ax=axs[2], format='%+2.0f dB')

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
    img3 = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', sr=sr,
                                    hop_length=512, ax=axs[3])
    axs[3].set_title('Chromagram (Melodic Spectrogram)')
    fig.colorbar(img3, ax=axs[3])

    plt.suptitle(f'Audio Analysis: {os.path.basename(filename)}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(advanced_viz_dir, f"{base_name}_complete_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def analyze_folder_features(input_folder, output_folder):
    """
    Analyze audio features across all files and generate summary visualizations.

    Parameters:
    -----------
    input_folder : str
        Path to folder containing audio files.
    output_folder : str
        Path to folder where summary visualizations will be saved.
    """
    summary_dir = os.path.join(output_folder, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.wav')]

    spectral_centroids = []
    spectral_rolloffs = []
    tempos = []
    file_names = []

    for filename in tqdm(audio_files, desc="Analyzing audio features"):
        try:
            y, sr = librosa.load(os.path.join(input_folder, filename))

            cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_centroids.append(np.mean(cent))

            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_rolloffs.append(np.mean(rolloff))


            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            tempos.append(tempo)

            file_names.append(os.path.splitext(filename)[0])

        except Exception as e:
            print(f"Error analyzing {filename}: {str(e)}")

    
    if file_names:
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(spectral_centroids)), spectral_centroids)
        plt.xticks(range(len(file_names)), file_names, rotation=90, fontsize=0.5)
        plt.xlabel('Audio Files')
        plt.ylabel('Spectral Centroid (Hz)')
        plt.title('Average Spectral Centroid (Audio Brightness) Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, 'spectral_centroids.png'), dpi=300)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(tempos)), tempos)
        plt.xticks(range(len(file_names)), file_names, rotation=90, fontsize=0.5)
        plt.xlabel('Audio Files')
        plt.ylabel('Tempo (BPM)')
        plt.title('Estimated Tempo Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, 'tempos.png'), dpi=300)
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.scatter(spectral_centroids, tempos)

        # for anotation
        
        # for i, file_name in enumerate(file_names):
        #     plt.annotate(file_name,
        #                  (spectral_centroids[i], tempos[i]),
        #                  fontsize=8,
        #                  alpha=0.7)

        plt.xlabel('Spectral Centroid (Hz)')
        plt.ylabel('Tempo (BPM)')
        plt.title('Spectral Centroid vs Tempo')
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, 'centroid_vs_tempo.png'), dpi=300)
        plt.close()


if __name__ == "__main__":
    
    input_folder = "../Data"
    output_folder = "./spectrograms"

    os.makedirs(output_folder, exist_ok=True)

    generator = SpectrogramGenerator(
        input_folder=input_folder,
        output_folder=output_folder,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmin=20,
        fmax=8000,
        sample_rate=None, # was 22050
        file_type=".wav"
    )

    generator.process_all_files(num_workers=4)

    sample_files = generator.audio_files[:]
    for filename in tqdm(sample_files):
        create_advanced_visualization(input_folder, output_folder, filename)

    analyze_folder_features(input_folder, output_folder)

    print("All processing completed!")