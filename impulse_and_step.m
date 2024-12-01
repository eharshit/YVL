% Unit Impulse Signal
n = -10:10; % Discrete time index
impulse = (n == 0); % Unit impulse at n = 0
figure;
stem(n, impulse, 'filled');
title('Unit Impulse Signal');
xlabel('n');
ylabel('Amplitude');
grid on;

% Unit Step Signal
unit_step = (n >= 0); % Unit step starts at n = 0
figure;
stem(n, unit_step, 'filled');
title('Unit Step Signal');
xlabel('n');
ylabel('Amplitude');
grid on;

% Sinusoidal Signal
n = 0:50; % Discrete time index
f = 0.1; % Frequency
sinusoid = sin(2 * pi * f * n); % Sinusoidal sequence
figure;
stem(n, sinusoid, 'filled');
title('Sinusoidal Signal');
xlabel('n');
ylabel('Amplitude');
grid on;

% Cosine Signal
cosine = cos(2 * pi * f * n); % Cosine sequence
figure;
stem(n, cosine, 'filled');
title('Cosine Signal');
xlabel('n');
ylabel('Amplitude');
grid on;

% Sawtooth Wave
f = 0.05; % Frequency
sawtooth_wave = sawtooth(2 * pi * f * n); % Sawtooth wave
figure;
stem(n, sawtooth_wave, 'filled');
title('Sawtooth Wave');
xlabel('n');
ylabel('Amplitude');
grid on;
