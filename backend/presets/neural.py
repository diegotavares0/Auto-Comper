"""
Neural tone refinement — spectrogram style transfer for guitar tone.

Uses neural style transfer on mel spectrograms:
- Content loss: preserve the musical content (notes, rhythm) of the input
- Style loss: match the timbral texture (gram matrix) of the reference
- Optimize output spectrogram via gradient descent
- Reconstruct audio via Griffin-Lim with phase from original

This is the "last 20%" that DSP can't capture — subtle nonlinear timbral
qualities, mic character, room interactions, compression color.
"""

import logging
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("comper")

# Lazy imports — only load heavy deps when neural is actually used
_torch = None
_torch_available = None


def is_available() -> bool:
    """Check if PyTorch is installed and neural refinement is available."""
    global _torch_available, _torch
    if _torch_available is None:
        try:
            import torch
            _torch = torch
            _torch_available = True
            log.info(f"  Neural refinement available (PyTorch {torch.__version__})")
        except ImportError:
            _torch_available = False
            log.info("  Neural refinement not available (PyTorch not installed)")
    return _torch_available


def refine_neural(
    audio: np.ndarray,
    sr: int,
    reference_audio: np.ndarray,
    intensity: float = 80.0,
    n_iterations: int = 200,
    content_weight: float = 1.0,
    style_weight: float = 1e4,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    progress_cb: Optional[Callable] = None,
) -> np.ndarray:
    """
    Apply neural spectrogram style transfer.

    Takes the DSP-processed audio and further refines it to match
    the reference's timbral texture using gradient-based optimization.

    Parameters
    ----------
    audio : np.ndarray
        Input audio (ideally already DSP-processed).
    sr : int
        Sample rate.
    reference_audio : np.ndarray
        Reference audio clip.
    intensity : float
        0-100% blending with original.
    n_iterations : int
        Optimization steps (more = closer match, slower).
    content_weight : float
        Weight for content preservation loss.
    style_weight : float
        Weight for style matching loss.

    Returns
    -------
    np.ndarray : Refined audio.
    """
    if not is_available():
        log.warning("  Neural not available, returning DSP-only result")
        return audio

    import torch
    import torch.nn.functional as F

    if progress_cb:
        progress_cb(78, "Iniciando refinamento neural...")

    device = torch.device("cpu")  # MPS/CUDA if available later

    # ── Convert to mel spectrograms ──
    def audio_to_mel(sig):
        """Convert audio to log-mel spectrogram tensor."""
        import librosa
        mel = librosa.feature.melspectrogram(
            y=sig, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, power=2.0,
        )
        log_mel = np.log1p(mel)
        return torch.from_numpy(log_mel).float().unsqueeze(0).unsqueeze(0).to(device)

    # Content: the input (what we want to preserve melodically)
    content_mel = audio_to_mel(audio)
    # Style: the reference (what we want to sound like)
    style_mel = audio_to_mel(reference_audio)

    # Target: start from content, optimize toward style
    target_mel = content_mel.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([target_mel], lr=0.01)

    def gram_matrix(x):
        """Compute Gram matrix for style loss."""
        b, c, h, w = x.size()
        features = x.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return G / (h * w)

    # Multi-scale style features (different frequency band groupings)
    def get_style_features(mel):
        """Extract multi-scale style features from mel spectrogram."""
        features = []
        # Full spectrogram
        features.append(gram_matrix(mel))
        # Low frequencies (bottom half of mel bands)
        features.append(gram_matrix(mel[:, :, :n_mels // 2, :]))
        # High frequencies (top half)
        features.append(gram_matrix(mel[:, :, n_mels // 2:, :]))
        return features

    style_grams = get_style_features(style_mel)

    if progress_cb:
        progress_cb(80, "Otimizando espectrograma...")

    # ── Optimization loop ──
    best_loss = float("inf")
    best_mel = target_mel.data.clone()

    for i in range(n_iterations):
        optimizer.zero_grad()

        # Content loss: preserve musical content
        content_loss = F.mse_loss(target_mel, content_mel) * content_weight

        # Style loss: match timbral texture
        target_grams = get_style_features(target_mel)
        style_loss = sum(
            F.mse_loss(tg, sg.detach())
            for tg, sg in zip(target_grams, style_grams)
        ) * style_weight

        # Total variation loss: smoothness
        tv_loss = (
            torch.mean(torch.abs(target_mel[:, :, 1:, :] - target_mel[:, :, :-1, :]))
            + torch.mean(torch.abs(target_mel[:, :, :, 1:] - target_mel[:, :, :, :-1]))
        ) * 1e-3

        total_loss = content_loss + style_loss + tv_loss
        total_loss.backward()
        optimizer.step()

        # Keep non-negative (it's a log-mel spectrogram)
        with torch.no_grad():
            target_mel.clamp_(min=0)

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_mel = target_mel.data.clone()

        if progress_cb and i % 50 == 0:
            pct = 80 + int(10 * i / n_iterations)
            progress_cb(
                pct,
                f"Refinando... iteracao {i}/{n_iterations} (loss={total_loss.item():.2f})"
            )

    if progress_cb:
        progress_cb(90, "Reconstruindo audio neural...")

    # ── Reconstruct audio from optimized mel spectrogram ──
    import librosa

    # Convert back from log-mel to mel
    optimized_mel = best_mel.squeeze().detach().cpu().numpy()
    mel_power = np.expm1(optimized_mel)

    # Use original phase for reconstruction (much better than Griffin-Lim)
    S_orig = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    phase = np.angle(S_orig)

    # Convert mel back to linear spectrogram
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    # Pseudo-inverse to go from mel back to linear
    mel_inv = np.linalg.pinv(mel_basis)
    S_linear = np.dot(mel_inv, mel_power)
    S_linear = np.maximum(S_linear, 0)

    # Match frame count to original
    min_frames = min(S_linear.shape[1], phase.shape[1])
    S_linear = S_linear[:, :min_frames]
    phase = phase[:, :min_frames]

    # Reconstruct with original phase
    S_complex = S_linear * np.exp(1j * phase)
    neural_audio = librosa.istft(S_complex, hop_length=hop_length, length=len(audio))

    # Blend with DSP result based on intensity
    blend = intensity / 100.0
    # Neural result blended on top of original (DSP-processed) audio
    # At 100% intensity: 50/50 DSP+Neural. At 50%: mostly DSP.
    neural_blend = blend * 0.5  # neural is always a refinement, not a replacement
    result = audio * (1 - neural_blend) + neural_audio * neural_blend

    # Safety: match original RMS to avoid level jumps
    orig_rms = np.sqrt(np.mean(audio ** 2))
    result_rms = np.sqrt(np.mean(result ** 2))
    if result_rms > 0:
        result = result * (orig_rms / result_rms)

    log.info(
        f"  Neural refinement: {n_iterations} iterations, "
        f"loss={best_loss:.2f}, blend={neural_blend:.0%}"
    )

    return result
