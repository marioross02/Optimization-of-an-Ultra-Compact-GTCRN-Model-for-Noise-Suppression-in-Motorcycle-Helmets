import os
import glob
import re
import tempfile
import subprocess
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pystoi import stoi
from gtcrn import GTCRN

# ==============================================================================
# --- CONFIGURAZIONE PARAMETRI SELEZIONABILI ---
# ==============================================================================                         
SPEAKERS = ['uomo1', 'uomo2', 'donna1', 'donna2']  # Lista degli speaker
# ==============================================================================

# --- PERCORSI ASSOLUTI DINAMICI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Scegli il percorso del checkpoint da utilizzare

#ckpt_path = os.path.join(BASE_DIR, 'checkpoints', 'model_trained_on_dns3.tar')
ckpt_path = os.path.join(BASE_DIR, 'checkpoints_finetuned', 'gtcrn_finetuned_best.tar')
#ckpt_path = os.path.join(BASE_DIR, 'checkpoints_finetuned_lossfornita', 'gtcrn_hybrid_finetuned_best.tar')

test_wavs_dir = os.path.join(BASE_DIR, 'test_wavs')
pesq_exe_path = os.path.join(BASE_DIR, 'pesq.exe')


def compute_pesq_native(clean_sig, deg_sig, fs=16000):
    """Calcola il PESQ ITU-T P.862 richiamando pesq.exe nativo."""
    if not os.path.exists(pesq_exe_path):
        raise FileNotFoundError(f"Impossibile trovare pesq.exe in: {pesq_exe_path}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        ref_path = os.path.join(tmp_dir, "ref.wav")
        deg_path = os.path.join(tmp_dir, "deg.wav")

        c_norm = (clean_sig / (np.max(np.abs(clean_sig)) + 1e-8) * 0.95)
        d_norm = (deg_sig / (np.max(np.abs(deg_sig)) + 1e-8) * 0.95)

        sf.write(ref_path, c_norm, fs, subtype='PCM_16')
        sf.write(deg_path, d_norm, fs, subtype='PCM_16')

        cmd = [pesq_exe_path, f"+{fs}", ref_path, deg_path]
        res = subprocess.run(cmd, capture_output=True, text=True)

        match = re.search(r"=\s*([\d\.]+)\s+([\d\.]+)", res.stdout)
        if match:
            return float(match.group(2))
        
        match_alt = re.search(r"=\s*([\d\.]+)", res.stdout)
        if match_alt:
            return float(match_alt.group(1))

    return 1.0


def compute_sisnr(clean, enh):
    """Calcolo del SI-SNR (Scale-Invariant Signal-to-Noise Ratio)."""
    clean = clean - np.mean(clean)
    enh = enh - np.mean(enh)
    alpha = np.dot(enh, clean) / (np.dot(clean, clean) + 1e-8)
    target = alpha * clean
    noise = enh - target
    return 10 * np.log10((np.dot(target, target) + 1e-8) / (np.dot(noise, noise) + 1e-8))


# --- 1. CARICAMENTO MODELLO (UNA SOLA VOLTA PRIMA DEL CICLO) ---
device = torch.device("cpu")
model = GTCRN().eval()

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint non trovato in: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model'])
window = torch.hann_window(512).pow(0.5)

# ==============================================================================
# CICLO DI ELABORAZIONE PER OGNI SPEAKER
# ==============================================================================
for SPEAKER in SPEAKERS:
    speaker_dir = os.path.join(test_wavs_dir, SPEAKER)
    clean_path = os.path.join(speaker_dir, f'clean_{SPEAKER}.wav')

    # Verifica presenza file clean
    if not os.path.exists(clean_path):
        print(f"\n[ATTENZIONE] File clean non trovato per '{SPEAKER}' in: {clean_path}. Skip...")
        continue

    clean_orig, fs = sf.read(clean_path, dtype='float32')
    assert fs == 16000, f"L'audio deve essere a 16kHz, trovato: {fs}Hz"

    mix_files = sorted(glob.glob(os.path.join(speaker_dir, 'mix_*.wav')))

    if not mix_files:
        print(f"\n[ATTENZIONE] Nessun file 'mix_*.wav' trovato per '{SPEAKER}'. Skip...")
        continue

    print(f"\n==================================================")
    print(f"=== AVVIO EVALUATION GTCRN - Speaker: {SPEAKER} ===")
    print(f"==================================================")
    print(f"Trovati {len(mix_files)} file da elaborare nella cartella.\n")

    metrics_list = []
    spectrogram_data = []  

    # --- 2. ELABORAZIONE AUDIO & METRICHE ---
    for idx, mix_path in enumerate(mix_files):
        filename = os.path.basename(mix_path)
        mix, _ = sf.read(mix_path, dtype='float32')

        # STFT
        stft_complex = torch.stft(
            torch.from_numpy(mix), 
            n_fft=512, hop_length=256, win_length=512, window=window, return_complex=True
        )
        input_stft = torch.view_as_real(stft_complex)

        # Forward pass GTCRN
        with torch.no_grad():
            output_stft = model(input_stft[None])[0]

        orig_len = len(mix)
        # ISTFT
        output_complex = torch.view_as_complex(output_stft.contiguous())
        enh = torch.istft(
            output_complex, 
            n_fft=512, 
            hop_length=256, 
            win_length=512, 
            window=window,
            length=orig_len
        )
        enh_gtcrn = enh.detach().cpu().numpy()

        # Allineamento
        min_len = min(len(clean_orig), len(mix), len(enh_gtcrn))
        clean_curr = clean_orig[:min_len]
        mix_curr = mix[:min_len]
        enh_curr = enh_gtcrn[:min_len]

        # Salvataggio audio elaborato
        enh_out_path = os.path.join(speaker_dir, f"enh_{filename}")
        sf.write(enh_out_path, enh_curr, fs)

        # Calcolo metriche
        try:
            p_noisy = compute_pesq_native(clean_curr, mix_curr, fs)
            p_gtcrn = compute_pesq_native(clean_curr, enh_curr, fs)
        except Exception as e:
            print(f"Errore PESQ su {filename}: {e}")
            p_noisy, p_gtcrn = 1.0, 1.0
        
        s_noisy = stoi(clean_curr, mix_curr, fs, extended=False)
        s_gtcrn = stoi(clean_curr, enh_curr, fs, extended=False)
        
        snr_noisy = compute_sisnr(clean_curr, mix_curr)
        snr_gtcrn = compute_sisnr(clean_curr, enh_curr)

        metrics_list.append({
            'filename': filename,
            'pesq_noisy': p_noisy, 'pesq_gtcrn': p_gtcrn,
            'stoi_noisy': s_noisy, 'stoi_gtcrn': s_gtcrn,
            'sisnr_noisy': snr_noisy, 'sisnr_gtcrn': snr_gtcrn
        })

        spectrogram_data.append({
            'filename': filename,
            'mix': mix_curr,
            'enh': enh_curr,
            'p_noisy': p_noisy, 'p_gtcrn': p_gtcrn,
            's_noisy': s_noisy, 's_gtcrn': s_gtcrn,
            'snr_noisy': snr_noisy, 'snr_gtcrn': snr_gtcrn
        })

        print(f"[{idx+1}/{len(mix_files)}] {filename} -> PESQ: {p_gtcrn:.2f} | STOI: {s_gtcrn:.3f} | SI-SNR: {snr_gtcrn:.2f} dB")

    # --- 3. GENERAZIONE IMMAGINE SPETTROGRAMMI ---
    num_files = len(spectrogram_data)
    fig, axes = plt.subplots(num_files, 2, figsize=(14, 3.2 * num_files), squeeze=False)

    for i, item in enumerate(spectrogram_data):
        axes[i, 0].specgram(item['mix'], Fs=fs, NFFT=512, noverlap=256, cmap='magma')
        axes[i, 0].set_title(
            f"NOISY: {item['filename']}\nPESQ: {item['p_noisy']:.2f} | STOI: {item['s_noisy']:.3f} | SI-SNR: {item['snr_noisy']:.2f} dB", 
            fontsize=9
        )
        axes[i, 0].set_ylabel("Freq (kHz)")
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_yticks([0, 2000, 4000, 6000, 8000])
        axes[i, 0].set_yticklabels(['0', '2', '4', '6', '8'])

        axes[i, 1].specgram(item['enh'], Fs=fs, NFFT=512, noverlap=256, cmap='magma')
        axes[i, 1].set_title(
            f"ENHANCED (GTCRN): {item['filename']}\nPESQ: {item['p_gtcrn']:.2f} | STOI: {item['s_gtcrn']:.3f} | SI-SNR: {item['snr_gtcrn']:.2f} dB", 
            fontsize=9
        )
        axes[i, 1].set_ylabel("Freq (kHz)")
        axes[i, 1].set_xlabel("Time (s)")
        axes[i, 1].set_yticks([0, 2000, 4000, 6000, 8000])
        axes[i, 1].set_yticklabels(['0', '2', '4', '6', '8'])

    plt.tight_layout()
    all_specs_path = os.path.join(speaker_dir, f'spettrogrammi_{SPEAKER}.png')
    plt.savefig(all_specs_path, dpi=300)
    plt.close()
    print(f"\n[OK] Spettrogrammi salvati in: {all_specs_path}")

    # --- 4. CALCOLO MEDIE E GENERAZIONE TABELLA ---
    avg_p_noisy = np.mean([m['pesq_noisy'] for m in metrics_list])
    avg_p_gtcrn = np.mean([m['pesq_gtcrn'] for m in metrics_list])
    avg_s_noisy = np.mean([m['stoi_noisy'] for m in metrics_list])
    avg_s_gtcrn = np.mean([m['stoi_gtcrn'] for m in metrics_list])
    avg_snr_noisy = np.mean([m['sisnr_noisy'] for m in metrics_list])
    avg_snr_gtcrn = np.mean([m['sisnr_gtcrn'] for m in metrics_list])

    table_data = [
        ["File", "PESQ (N)", "PESQ (GT)", "STOI (N)", "STOI (GT)", "SI-SNR N (dB)", "SI-SNR GT (dB)"]
    ]

    for m in metrics_list:
        table_data.append([
            m['filename'],
            f"{m['pesq_noisy']:.2f}", f"{m['pesq_gtcrn']:.2f}",
            f"{m['stoi_noisy']:.3f}", f"{m['stoi_gtcrn']:.3f}",
            f"{m['sisnr_noisy']:.2f}", f"{m['sisnr_gtcrn']:.2f}"
        ])

    table_data.append([
        "MEDIA",
        f"{avg_p_noisy:.2f}", f"{avg_p_gtcrn:.2f}",
        f"{avg_s_noisy:.3f}", f"{avg_s_gtcrn:.3f}",
        f"{avg_snr_noisy:.2f}", f"{avg_snr_gtcrn:.2f}"
    ])

    fig_height = max(4, len(table_data) * 0.45)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for j in range(len(table_data[0])):
        table[(0, j)].get_text().set_weight('bold')
        table[(0, j)].set_facecolor('#d9e1f2')
        table[(len(table_data)-1, j)].get_text().set_weight('bold')
        table[(len(table_data)-1, j)].set_facecolor('#e2efda')

    plt.title(f"Risultati Speaker: {SPEAKER}", fontsize=12, fontweight='bold', pad=20)
    table_img_path = os.path.join(speaker_dir, f'tabella_risultati_{SPEAKER}.png')
    plt.savefig(table_img_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Tabella risultati salvata in: {table_img_path}\n")