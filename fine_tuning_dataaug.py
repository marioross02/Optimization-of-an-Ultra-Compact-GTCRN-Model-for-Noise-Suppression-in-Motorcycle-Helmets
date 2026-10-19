import glob
import os
import random
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.functional as F
from audiomentations import Compose, Gain, PitchShift, TimeStretch
from torch.utils.data import DataLoader, Dataset

from gtcrn import GTCRN


# =============================================================================
# 0. SEED WORKER PER RENDER DETERMINISTICI I WORKER DEL DATALOADER
# =============================================================================
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# 1. DATASET CON DATA AUGMENTATION, MIXING DINAMICO E SEED ISOLATO
# =============================================================================
class DynamicAudioDataset(Dataset):

    def __init__(
        self,
        clean_dir,
        noise_dir,
        segment_len_sec=3.0,
        sample_rate=16000,
        snr_range=(-8, 3),
        augment=True,
        seed=None,
        name="Dataset",
    ):
        # Ricerca estesa per .wav e .flac (sia minuscoli che maiuscoli, ricorsivi)
        self.clean_files = sorted(
            list(
                set(
                    glob.glob(os.path.join(clean_dir, "*.wav"))
                    + glob.glob(os.path.join(clean_dir, "*.WAV"))
                    + glob.glob(os.path.join(clean_dir, "*.flac"))
                    + glob.glob(os.path.join(clean_dir, "*.FLAC"))
                    + glob.glob(os.path.join(clean_dir, "**/*.wav"), recursive=True)
                    + glob.glob(os.path.join(clean_dir, "**/*.WAV"), recursive=True)
                    + glob.glob(os.path.join(clean_dir, "**/*.flac"), recursive=True)
                    + glob.glob(os.path.join(clean_dir, "**/*.FLAC"), recursive=True)
                )
            )
        )
        self.noise_files = sorted(
            list(
                set(
                    glob.glob(os.path.join(noise_dir, "*.wav"))
                    + glob.glob(os.path.join(noise_dir, "*.WAV"))
                    + glob.glob(os.path.join(noise_dir, "*.flac"))
                    + glob.glob(os.path.join(noise_dir, "*.FLAC"))
                    + glob.glob(os.path.join(noise_dir, "**/*.wav"), recursive=True)
                    + glob.glob(os.path.join(noise_dir, "**/*.WAV"), recursive=True)
                    + glob.glob(os.path.join(noise_dir, "**/*.flac"), recursive=True)
                    + glob.glob(os.path.join(noise_dir, "**/*.FLAC"), recursive=True)
                )
            )
        )

        # Stampa di controllo per verificare quanti file ha preso in ingresso
        print(f"[{name}] -> Puliti caricati: {len(self.clean_files)} da {clean_dir}")
        print(f"[{name}] -> Rumore caricati: {len(self.noise_files)} da {noise_dir}")

        self.segment_len = int(segment_len_sec * sample_rate)
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.augment = augment
        self.seed = seed

        if len(self.clean_files) == 0:
            raise RuntimeError(
                f"Nessun file audio trovato nella cartella clean: {clean_dir}"
            )
        if len(self.noise_files) == 0:
            raise RuntimeError(
                f"Nessun file audio trovato nella cartella noise: {noise_dir}"
            )

        if self.augment:
            self.transform = Compose([
                PitchShift(min_semitones=-0.8, max_semitones=0.8, p=0.5),
                TimeStretch(min_rate=0.92, max_rate=1.08, leave_length_unchanged=False, p=0.5),
                Gain(min_gain_db=-3.0, max_gain_db=3.0, p=0.5)
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.clean_files)

    def _load_and_crop(self, filepath, apply_aug=False, rng=random):
        data, sr = sf.read(filepath, dtype="float32")

        if data.ndim > 1:
            data = data[:, 0]

        if apply_aug and self.transform is not None:
            data = self.transform(samples=data, sample_rate=sr)

        audio = torch.from_numpy(data).unsqueeze(0)

        if sr != self.sample_rate:
            audio = F.resample(
                audio, orig_freq=sr, new_freq=self.sample_rate
            )

        num_samples = audio.shape[-1]
        if num_samples < self.segment_len:
            pad_len = self.segment_len - num_samples
            audio = nn.functional.pad(audio, (0, pad_len))
        elif num_samples > self.segment_len:
            start = rng.randint(0, num_samples - self.segment_len - 1)
            audio = audio[:, start : start + self.segment_len]

        return audio.squeeze(0)

    def __getitem__(self, idx):
        if self.seed is not None:
            rng = random.Random(self.seed + idx)
        else:
            rng = random

        clean = self._load_and_crop(self.clean_files[idx], apply_aug=self.augment, rng=rng)
        
        noise_path = rng.choice(self.noise_files)
        noise = self._load_and_crop(noise_path, apply_aug=False, rng=rng)

        snr_db = rng.uniform(*self.snr_range)
        clean_power = torch.mean(clean**2) + 1e-12
        noise_power = torch.mean(noise**2) + 1e-12

        desired_noise_power = clean_power / (10 ** (snr_db / 10.0))
        scale = torch.sqrt(desired_noise_power / noise_power)

        noisy = clean + noise * scale
        return noisy, clean


# =============================================================================
# 2. FUNZIONE DI LOSS GTCRN (ALPHA = 0.05, BETA = 0.2)
# =============================================================================
class GTCRNLoss(nn.Module):

    def __init__(self, n_fft=512, hop_length=256, alpha=0.05, beta=0.2):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha = alpha
        self.beta = beta

        window = torch.hann_window(self.n_fft).pow(0.5)
        self.register_buffer("window", window)

    def _si_snr(self, est, target, eps=1e-8):
        target = target - torch.mean(target, dim=-1, keepdim=True)
        est = est - torch.mean(est, dim=-1, keepdim=True)

        s_target = (torch.sum(est * target, dim=-1, keepdim=True) * target) / (
            torch.sum(target**2, dim=-1, keepdim=True) + eps
        )
        e_noise = est - s_target

        si_snr = 10 * torch.log10(
            (torch.sum(s_target**2, dim=-1) + eps)
            / (torch.sum(e_noise**2, dim=-1) + eps)
        )
        return -torch.mean(si_snr)

    def forward(self, est_wav, target_wav, eps=1e-8):
        loss_sisnr = self._si_snr(est_wav, target_wav)

        spec_est = torch.stft(
            est_wav,
            self.n_fft,
            self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )
        spec_target = torch.stft(
            target_wav,
            self.n_fft,
            self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )

        mag_est = torch.abs(spec_est) + eps
        mag_target = torch.abs(spec_target) + eps

        loss_mag = nn.functional.mse_loss(mag_est**0.3, mag_target**0.3)

        real_est_compressed = spec_est.real / (mag_est**0.7)
        real_target_compressed = spec_target.real / (mag_target**0.7)
        loss_real = nn.functional.mse_loss(
            real_est_compressed, real_target_compressed
        )

        imag_est_compressed = spec_est.imag / (mag_est**0.7)
        imag_target_compressed = spec_target.imag / (mag_target**0.7)
        loss_imag = nn.functional.mse_loss(
            imag_est_compressed, imag_target_compressed
        )

        total_loss = (
            self.alpha * loss_sisnr
            + (1 - self.beta) * loss_mag
            + self.beta * (loss_real + loss_imag)
        )

        return total_loss


# =============================================================================
# 3. WRAPPER MODELLO GTCRN
# =============================================================================
class GTCRNPipeline(nn.Module):

    def __init__(self, model, n_fft=512, hop_length=256):
        super().__init__()
        self.model = model
        self.n_fft = n_fft
        self.hop_length = hop_length

        window = torch.hann_window(self.n_fft).pow(0.5)
        self.register_buffer("window", window)

    def forward(self, noisy_wav):
        spec_complex = torch.stft(
            noisy_wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )
        spec_real = torch.view_as_real(spec_complex)

        spec_enh_real = self.model(spec_real)

        spec_enh_complex = torch.view_as_complex(spec_enh_real.contiguous())
        wav_enh = torch.istft(
            spec_enh_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            length=noisy_wav.shape[-1],
        )
        return wav_enh


# =============================================================================
# 4. MAIN PIPELINE DI FINE-TUNING
# =============================================================================
def train_gtcrn_finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo in uso: {device}")

    batch_size = 8
    epochs = 300
    learning_rate = 1e-3

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    train_clean_dir = os.path.join(BASE_DIR, "dataset", "train", "clean")
    train_noise_dir = os.path.join(BASE_DIR, "dataset", "train", "noise")
    val_clean_dir = os.path.join(BASE_DIR, "dataset", "val", "clean")
    val_noise_dir = os.path.join(BASE_DIR, "dataset", "val", "noise")

    #ckpt_input_path = os.path.join(
    #    BASE_DIR, "checkpoints", "model_trained_on_dns3.tar")

    ckpt_input_path = os.path.join(BASE_DIR, "altro", "model_trained_on_dns3.tar")   #cartella fittizia per non caricare i pesi pre-addestrati, così da testare il fine-tuning da zero
    ckpt_output_dir = os.path.join(BASE_DIR, "checkpoints_finetuned")

    for path in [
        train_clean_dir,
        train_noise_dir,
        val_clean_dir,
        val_noise_dir,
        ckpt_output_dir,
    ]:
        os.makedirs(path, exist_ok=True)

    g_val = torch.Generator()
    g_val.manual_seed(42)

    print("=" * 60)
    print("CONTROLLO CARICAMENTO FILE DAL DISCO:")
    print("=" * 60)

    try:
        train_dataset = DynamicAudioDataset(train_clean_dir, train_noise_dir, augment=True, seed=None, name="TRAIN")
        val_dataset = DynamicAudioDataset(val_clean_dir, val_noise_dir, augment=False, seed=42, name="VAL")

    except RuntimeError as e:
        print(f"\n[ERRORE DATASET]: {e}")
        return
    print("=" * 60)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g_val,
        pin_memory=(device.type == "cuda"),
    )

    base_model = GTCRN()
    pipeline = GTCRNPipeline(base_model).to(device)

    if os.path.exists(ckpt_input_path):
        ckpt = torch.load(ckpt_input_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            base_model.load_state_dict(ckpt["model"])
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            base_model.load_state_dict(ckpt["state_dict"])
        else:
            base_model.load_state_dict(ckpt)
        print(
            f"Pesi pre-addestrati caricati con successo da: {ckpt_input_path}"
        )
    else:
        print(
            f"ATTENZIONE: Checkpoint non trovato in {ckpt_input_path}. Avvio"
            " addestramento da zero."
        )
    # =========================================================================
    # CONGELAMENTO PARZIALE (TEST A: Encoder e ERB bloccati, DPGRNN + Decoder liberi)
    # =========================================================================
    
    # 1. Congeliamo tutto il modello di base
    #for param in base_model.parameters():
    #    param.requires_grad = False

    # 2. Sblocchiamo SOLO i blocchi DPGRNN e il Decoder
    #for param in base_model.dpgrnn1.parameters():
    #    param.requires_grad = True
    #for param in base_model.dpgrnn2.parameters():
    #    param.requires_grad = True
    #for param in base_model.decoder.parameters():
    #    param.requires_grad = True

    # (Opzionale ma utilissimo) Stampiamo il conteggio per verificare che i layer giusti siano sbloccati
    #trainable_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    #total_params = sum(p.numel() for p in pipeline.parameters())
    #print(f"\n[CONFIGURAZIONE PARAMETRI]:")
    #print(f" -> Parametri Totali: {total_params:,}")
    #print(f" -> Parametri Addestrabili (DPGRNN + Decoder): {trainable_params:,}")
    #print(f" -> Parametri Congelati (Encoder + ERB): {total_params - trainable_params:,}\n")

    # 3. L'Optimizer deve filtrare e considerare SOLO i parametri con requires_grad=True
    #optimizer = torch.optim.AdamW(
    #    filter(lambda p: p.requires_grad, pipeline.parameters()), 
    #    lr=learning_rate, 
    #    weight_decay=1e-4
    #)

    optimizer = torch.optim.AdamW(
        pipeline.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-4
    )

    criterion = GTCRNLoss(alpha=0.05, beta=0.2).to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_val_loss = float("inf")
    early_stop_patience = 20
    no_improve_epochs = 0

    print(
        "\n--- Inizio Fine-Tuning GTCRN ( Data Aug, CosineAnnealingLR, LR=1e-3,"
        " alpha=0.05, beta=0.2) ---"
    )
    for epoch in range(1, epochs + 1):
        pipeline.train()
        train_loss = 0.0

        for batch_idx, (noisy, clean) in enumerate(train_loader):
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            enhanced = pipeline(noisy)
            loss = criterion(enhanced, clean)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                pipeline.parameters(), max_norm=1.0
            )

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        pipeline.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                enhanced = pipeline(noisy)
                loss = criterion(enhanced, clean)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(
            f"Epoca [{epoch:03d}/{epochs:03d}] | LR: {current_lr:.2e} | Train"
            f" Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            save_path = os.path.join(
                ckpt_output_dir, "gtcrn_finetuned_best.tar"
            )

            torch.save(
                {
                    "model": base_model.state_dict(),
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "val_loss": avg_val_loss,
                },
                save_path,
            )
            print(f" ---> [Miglior Modello Salvato]: {save_path}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                print(
                    f"\n[EARLY STOPPING]: La Val Loss non è migliorata per"
                    f" {early_stop_patience} epoche. Interruzione."
                )
                break


if __name__ == "__main__":
    train_gtcrn_finetune()