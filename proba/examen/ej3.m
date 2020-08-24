pkg load statistics
% Setear la semilla
rand ("seed", 123);

% N simulaciones
N = 200

% cantidad de muestras
n = 10

% medicion, la media
mu = 48

% dispersion
sigma = 4

% intervalo 95%, alfa:
alfa = 1 - 0.95
z = norminv(1 - alfa/2)

% el intervalo de confianza de 95% teorico es:
mu_min_teorico = mu - z*sigma/sqrt(n);
mu_max_teorico = mu + z*sigma/sqrt(n);

mu_h0 = 45;
z_h0 = (mu - mu_h0) / (sigma/sqrt(n))

if(z >= z_h0)
  disp(strcat(num2str(z_h0), ' <= ', num2str(z))) 
  disp('no hay suficiente evidencia, mu_h0 cae dentro del intervalo, H0 aceptada!')
else
  disp(strcat(num2str(z_h0), ' > ', num2str(z)))
  disp('hay suficiente evidencia, mu_h0 cae fuera del intervalo, H0 rechazada!')
end


% M ensayos de N ensayos
M = 3;
ensayos = zeros(M,1);
ensayos(1) = 50;
ensayos(2) = 100;
ensayos(3) = 1000;

error_intervalo_simulado_vector = zeros(M,1)

for e = 1:M
  % N ensayos
  N = ensayos(e)

  % nivel de confianza del intervalo
  intervalo = 0;

  for i=1:N

      % generar distribucion
      X = sigma/sqrt(n)*randn(n,1) + mu;

      % se verifica cuantos valores están dentro del intervalo teórico
      intervalo = intervalo + (1/(N*n)) *sum(X>=mu_min_teorico & X<=mu_max_teorico);

  end

  % estimacion del intervalo de confianza
  intervalo_teorico = 1 - alfa
  intervalo_simulado = intervalo
  error_intervalo_simulado = (abs(intervalo_simulado-intervalo_teorico)/intervalo_teorico)*100
  error_intervalo_simulado_vector(e) = error_intervalo_simulado;
end

figure()
plot(ensayos, error_intervalo_simulado_vector)
ylabel ("error [%]");
xlabel ("N muestras");
title ("Error de intervalo simulado")
