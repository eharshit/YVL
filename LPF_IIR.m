clc;
clear all;
close all;

disp('Enter the IIR filter design specifications:');

% Input specifications
rp = input('Enter the passband ripple (dB): ');
rs = input('Enter the stopband ripple (dB): ');
wp = input('Enter the passband frequency (Hz): ');
ws = input('Enter the stopband frequency (Hz): ');
fs = input('Enter the sampling frequency (Hz): ');

% Normalize frequencies
w1 = 2 * wp / fs; % Normalized passband frequency
w2 = 2 * ws / fs; % Normalized stopband frequency

% Determine filter order and cutoff frequency
[n, wn] = buttord(w1, w2, rp, rs, 's');

disp('Frequency response of IIR HPF is:');

% High-pass Butterworth filter design
[b, a] = butter(n, wn, 'high', 's');

% Frequency response
w = 0:0.01:pi; % Frequency range for plotting
[h, om] = freqs(b, a, w); % Frequency response
m = 20 * log10(abs(h)); % Magnitude in dB
an = angle(h); % Phase response

% Plot magnitude response
figure;
subplot(2, 1, 1);
plot(om/pi, m, 'LineWidth', 1.5);
title('Magnitude Response of IIR Filter');
xlabel('Normalized Frequency (\times\pi rad/sample)');
ylabel('Gain (dB)');
grid on;

% Plot phase response
subplot(2, 1, 2);
plot(om/pi, an, 'LineWidth', 1.5);
title('Phase Response of IIR Filter');
xlabel('Normalized Frequency (\times\pi rad/sample)');
ylabel('Phase (radians)');
grid on;
