import numpy as np
import librosa
import os
import scipy.io.wavfile
import soundfile as sf
import scipy.stats
from math import floor


def extract(amps, sample_rate=44100):

  window_size_ms = 23.2
  n_fft = int(floor((sample_rate * window_size_ms) / 500.) * 2)

  # 50% frame overlap
  hop_length = int(n_fft / 2)
  mfcc_mat = librosa.feature.mfcc(
      y=amps, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=40)

  metric_min = np.min(mfcc_mat[0:25, :], axis=1)
  metric_max = np.max(mfcc_mat[0:25, :], axis=1)
  metric_median = np.median(mfcc_mat[0:25, :], axis=1)
  metric_mean = np.mean(mfcc_mat[0:25, :], axis=1)
  metric_variance = np.var(mfcc_mat[0:25, :], axis=1)
  metric_skewness = scipy.stats.skew(mfcc_mat[0:25, :], axis=1)
  metric_kurtosis = scipy.stats.kurtosis(mfcc_mat[0:25, :], axis=1)
  # delta
  max_width = 9  # this is the default value in librosa.feature.delta, I'm unsure what is optimal
  input_length = mfcc_mat.shape[1]
  width = input_length if input_length < max_width else max_width
  mfcc_delta = librosa.feature.delta(mfcc_mat[0:25, :], axis=1, width=7)
  metric_mean_delta = np.mean(mfcc_delta[0:25, :], axis=1)
  metric_variance_delta = np.var(mfcc_delta[0:25, :], axis=1)
  # delta-delta
  mfcc_delta_2 = librosa.feature.delta(
      mfcc_mat[0:25, :], axis=1, order=2, width=7)
  metric_mean_delta_2 = np.mean(mfcc_delta_2[0:25, :], axis=1)
  metric_variance_delta_2 = np.var(mfcc_delta_2[0:25, :], axis=1)

  feature_vector = np.hstack((
      metric_min,              # 1
      metric_max,              # 2
      metric_median,           # 3
      metric_mean,             # 4
      metric_variance,         # 5
      metric_skewness,         # 6
      metric_kurtosis,         # 7
      metric_mean_delta,       # 8
      metric_variance_delta,   # 9
      metric_mean_delta_2,     # 10
      metric_variance_delta_2  # 11
  ))

  return feature_vector.tolist()
