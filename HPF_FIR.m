clc;
clear; 
close all;

% Parameters
n = 20;          % Filter order
fp = 300;        % Passband frequency
fq = 200;        % Unused parameter, consider removing
fs = 1000;       % Sampling frequency
fn = 2 * fp / fs; % Normalized cutoff frequency

% Blackman window and high-pass filter design
window = blackman(n + 1); % Blackman window
b = fir1(n, fn, 'high', window); % High-pass FIR filter

% Frequency response
[H, W] = freqz(b, 1, 128);

% Magnitude response
subplot(2, 1, 1);
plot(W/pi,abs(H)); % Gain in dB
title('Magnitude Response of HPF');
ylabel('Gain (dB)');
xlabel('Normalized Frequency (\times\pi rad/sample)');
grid on;

% Phase response
subplot(2, 1, 2);
plot(W/pi, angle(H)); % Phase in radians
title('Phase Response of HPF');
ylabel('Phase (radians)');
xlabel('Normalized Frequency (\times\pi rad/sample)');
grid on;
