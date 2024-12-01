n = -20:20;
step_signal = (n >= 0);
figure;
stem(n, step_signal, 'filled'); 
title('Step Signal');
xlabel('n');
ylabel('u[n]');
grid on;

A = 1;       
F = 0.1;     
phi = 0;      
sine_signal = A * sin(2 * pi * F * n + phi);
figure;
stem(n, sine_signal, 'filled'); 
title('Sine Signal');
xlabel('n');
ylabel('sin(2 \pi F n + \phi)');
grid on;

cosine_signal = A * cos(2 * pi * F * n + phi);
figure;
stem(n, cosine_signal, 'filled'); 
title('Cosine Signal');
xlabel('n');
ylabel('cos(2 \pi F n + \phi)');
grid on;