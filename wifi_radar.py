#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import struct
import logging
import argparse
import csv
import joblib
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from typing import Tuple, Dict, Any
from FeatureExtractor import FeatureExtractor

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class Config:
    def __init__(self,
                 csi_file: str = 'csi_data.bin',
                 subcarriers: int = 30,
                 filter_kernel_size: int = 3,
                 export_csv: bool = True,
                 output_csv_file: str = 'csi_results.csv',
                 plot_figures: bool = True,
                 save_figures: bool = False,
                 figures_dir: str = 'figures',
                 buffer_size: int = 100):
        self.csi_file = csi_file
        self.subcarriers = subcarriers
        self.filter_kernel_size = filter_kernel_size
        self.export_csv = export_csv
        self.output_csv_file = output_csv_file
        self.plot_figures = plot_figures
        self.save_figures = save_figures
        self.figures_dir = figures_dir
        self.buffer_size = buffer_size

        if self.save_figures and not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir, exist_ok=True)

class CSIReader:
    def __init__(self, config: Config):
        self.config = config

    def read_new_csi_data(self, file_handle, current_position: int) -> Tuple[np.ndarray, int]:
        file_handle.seek(current_position)
        bytes_per_amostra = self.config.subcarriers * 2 * 2
        bytes_read = file_handle.read(bytes_per_amostra)
        if not bytes_read or len(bytes_read) < bytes_per_amostra:
            return np.array([]), current_position
        csi_vals = struct.unpack(f'{self.config.subcarriers*2}h', bytes_read)
        complex_csi = np.array(csi_vals[0::2]) + 1j*np.array(csi_vals[1::2])
        current_position = file_handle.tell()
        return complex_csi, current_position

class CSIFilter:
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size

    def denoise_amplitude(self, csi: np.ndarray) -> np.ndarray:
        amplitudes = np.abs(csi)
        filtered = medfilt(amplitudes, kernel_size=self.kernel_size)
        return filtered

    def unwrap_phase(self, csi: np.ndarray) -> np.ndarray:
        phases = np.angle(csi)
        unwrapped = np.unwrap(phases)
        return unwrapped

class CSIAnalyzer:
    def __init__(self, config: Config):
        self.config = config

    def compute_mean_amplitude(self, csi_amplitude: np.ndarray) -> float:
        return np.mean(csi_amplitude)

    def compute_mean_phase(self, csi_phase: np.ndarray) -> float:
        return np.mean(csi_phase)

    def compute_frequency_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        freq_domain = np.fft.fftshift(np.fft.fft(signal))
        freqs = np.fft.fftfreq(len(signal))
        return freqs, np.abs(freq_domain)

    def plot_time_series(self, time_signal: np.ndarray, title: str, ylabel: str, filename: str):
        plt.figure(figsize=(8,4))
        plt.plot(time_signal)
        plt.title(title)
        plt.xlabel("Sample")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        if self.config.plot_figures:
            plt.show()
        if self.config.save_figures:
            plt.savefig(os.path.join(self.config.figures_dir, filename))
        plt.close()

    def plot_frequency_spectrum(self, freqs: np.ndarray, spectrum: np.ndarray, title: str, filename: str):
        plt.figure(figsize=(8,4))
        plt.plot(freqs, spectrum)
        plt.title(title)
        plt.xlabel("Normalized Frequency")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.tight_layout()
        if self.config.plot_figures:
            plt.show()
        if self.config.save_figures:
            plt.savefig(os.path.join(self.config.figures_dir, filename))
        plt.close()

class ResultsExporter:
    def __init__(self, config: Config):
        self.config = config

    def export_to_csv(self, data: Dict[str, Any]):
        if not self.config.export_csv:
            logging.info("Export CSV disabled. Skipping this step.")
            return
        file_exists = os.path.isfile(self.config.output_csv_file)
        with open(self.config.output_csv_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(data.keys())
            writer.writerow(data.values())
        logging.info(f"Exported data to {self.config.output_csv_file}")

def main(config: Config):
    reader = CSIReader(config)
    csi_filter = CSIFilter(kernel_size=config.filter_kernel_size)
    analyzer = CSIAnalyzer(config)
    exporter = ResultsExporter(config)
    buffer = []
    current_position = 0

    try:
        with open(config.csi_file, 'rb') as f:
            while True:
                new_csi, current_position = reader.read_new_csi_data(f, current_position)
                if new_csi.size > 0:
                    amplitude_filtered = csi_filter.denoise_amplitude(new_csi)
                    phase_unwrapped = csi_filter.unwrap_phase(new_csi)
                    mean_amp = analyzer.compute_mean_amplitude(amplitude_filtered)
                    mean_phase = analyzer.compute_mean_phase(phase_unwrapped)
                    buffer.append((mean_amp, mean_phase))
                    if len(buffer) >= config.buffer_size:
                        mean_amplitudes, mean_phases = zip(*buffer)
                        mean_amplitudes = np.array(mean_amplitudes)
                        mean_phases = np.array(mean_phases)
                        features = FeatureExtractor.extract_features(mean_amplitudes, mean_phases).reshape(1, -1)
                        features_scaled = scaler.transform(features)
                        prediction = model.predict(features_scaled)
                        logging.info(f"Prediction: {prediction[0]}")
                        exporter.export_to_csv({
                            'mean_amplitude': mean_amp,
                            'mean_phase': mean_phase,
                            'prediction': prediction[0]
                        })
                        buffer = []
                time.sleep(0.5)
    except KeyboardInterrupt:
        logging.info("Real-time processing stopped by user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wi-Fi Radar Prototype with Real-Time ML Predictions")
    parser.add_argument("--file", type=str, default="csi_data.bin", help="Path to CSI data file.")
    parser.add_argument("--subcarriers", type=int, default=30, help="Number of subcarriers.")
    parser.add_argument("--kernel", type=int, default=3, help="Size of the median filter kernel.")
    parser.add_argument("--no-export", action="store_true", help="Disable exporting results to CSV.")
    parser.add_argument("--csv-file", type=str, default="csi_results.csv", help="Name of the output CSV file.")
    parser.add_argument("--no-plot", action="store_true", help="Do not display plots on screen.")
    parser.add_argument("--save-fig", action="store_true", help="Save plots to files instead of displaying.")
    parser.add_argument("--fig-dir", type=str, default="figures", help="Directory to save figures.")
    parser.add_argument("--buffer-size", type=int, default=100, help="Number of samples to buffer before prediction.")

    args = parser.parse_args()

    config = Config(
        csi_file=args.file,
        subcarriers=args.subcarriers,
        filter_kernel_size=args.kernel,
        export_csv=not args.no_export,
        output_csv_file=args.csv_file,
        plot_figures=not args.no_plot,
        save_figures=args.save_fig,
        figures_dir=args.fig_dir,
        buffer_size=args.buffer_size
    )

    main(config)
