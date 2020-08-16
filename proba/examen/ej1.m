pkg load statistics
% Setear la semilla
rand ("seed", 123);

n = 68;
p = 0.01;
k = 1;

% Probabilidad de encontrar 1 o más fósforos defectuosos
% para N fósforos 
prob = 1 - nchoosek(n,0) * p^0 * (1-p)^(n-0)


% media teorica
media_teorica = n*p

% desvio teorico
desvio_teorico = sqrt(n*p*(1-p))

% M ensayos de N ensayos
M = 3;
ensayos = zeros(M,1);
ensayos(1) = 100;
ensayos(2) = 1000;
ensayos(3) = 10000;

error_media_muestral_vector = zeros(M,1);
error_desvio_muestral_vector = zeros(M,1);

for e = 1:M
  % N ensayos
  N = ensayos(e)

  % numero de fósforos defectuosos
  % 1 --> defectuoso
  % 0 --> no defectuoso
  fosforos_defectuosos = zeros(N,1);


  for i = 1:N

    for j = 1:n

      if(rand() < p)
        fosforos_defectuosos(i) = fosforos_defectuosos(i) + 1;
      end

    end

  end

  % calcular la media y varianza
  media_muestral = sum(fosforos_defectuosos) / N

  % desvio muestral
  desvio_muestral = sqrt((1/(N-1)) * sum((fosforos_defectuosos - media_muestral * ones(N,1)).^2))

  error_media_muestral = (abs(media_muestral - media_teorica) / media_teorica) * 100
  error_media_muestral_vector(e) = error_media_muestral;
  
  error_desvio_muestral = (abs(desvio_muestral - desvio_teorico) / desvio_teorico) * 100
  error_desvio_muestral_vector(e) = error_desvio_muestral;

end

figure()
plot(ensayos, error_media_muestral_vector)
ylabel ("error [%]");
xlabel ("N muestras");
title ("Error de la media muestral")

figure()
plot(ensayos, error_varianza_muestral_vector)
ylabel ("error [%]");
xlabel ("N muestras");
title ("Error del desvio muestral")

