fs = 8000;
t = 0:1/fs:0.5-1/fs; 
f1 = 770;
f2 = 1336;
dtmf_signal = cos(2*pi*f1*t) + cos(2*pi*f2*t);
plot(t, dtmf_signal);
xlabel('Time (s)');
ylabel('Amplitude');
title('DTMF Signal for Digit 5');
sound(dtmf_signal, fs);