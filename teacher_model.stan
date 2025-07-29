data {
  int<lower = 1> N;                  // number of students
  int<lower = 1> J;                  // number of unique teachers
  array[N] int<lower = 1, upper = J> teacher_8;
  array[N] int<lower = 1, upper = J> teacher_9;
  array[N] int<lower = 1, upper = J> teacher_10;

  vector<lower = 0, upper = 1>[N] x8;    // grade 8
  vector<lower = 0, upper = 1>[N] x9;    // grade 9
  vector<lower = 0, upper = 1>[N] y10;   // grade 10
}

transformed data {
  real mean_x8 = mean(x8);
  real mean_x9 = mean(x9);
}

parameters {
  // Latent teacher bias
  vector[J] zeta;
  
  // Grade-specific teacher effects
  vector[J] tau_8;
  vector[J] tau_9;
  vector[J] tau_10;

  // Model parameters
  real alpha8;
  real alpha9;
  real alpha10;
  real<lower = 0> beta1;
  real<lower = 0> beta2;
  real<lower = 0> beta3;

  // Dispersion
  real<lower = 0> phi8;
  real<lower = 0> phi9;
  real<lower = 0> phi10;
}

transformed parameters {
  vector[N] mu8;
  vector[N] mu9;
  vector[N] mu10;

  for (i in 1:N) {
    mu8[i] = inv_logit(tau_8[teacher_8[i]] + alpha8);
    mu9[i] = inv_logit(tau_9[teacher_9[i]] + alpha9 + beta1 * (x8[i] - mean_x8));
    mu10[i] = inv_logit(tau_10[teacher_10[i]] + alpha10 + beta2 * (x8[i] - mean_x8) + beta3 * (x9[i] - mean_x9));
  }
}

model {
  // Priors
  alpha8 ~ normal(1.4, 0.03);
  alpha9 ~ normal(1.2, 0.03);
  alpha10 ~ normal(0.95, 0.03);
  beta1 ~ normal(5, 1.5);
  beta2 ~ normal(2, 1);
  beta3 ~ normal(4, 2);
  phi8 ~ normal(10, 3);
  phi9 ~ normal(10, 3);
  phi10 ~ normal(10, 3);
  zeta ~ normal(0, 0.3);
  tau_8 ~ normal(zeta, 0.1);
  tau_9 ~ normal(zeta, 0.1);
  tau_10 ~ normal(zeta, 0.1);

  // Likelihood
  x8 ~ beta(mu8 .* phi8, (1 - mu8) .* phi8);
  x9 ~ beta(mu9 .* phi9, (1 - mu9) .* phi9);
  y10 ~ beta(mu10 .* phi10, (1 - mu10) .* phi10);
}

generated quantities {
  vector[N] x9_hat;
  vector[N] y10_hat;
  vector[N] log_lik;

  for (i in 1:N) {
    x9_hat[i] = beta_rng(mu9[i] * phi9, (1 - mu9[i]) * phi9);
    y10_hat[i] = beta_rng(mu10[i] * phi10, (1 - mu10[i]) * phi10);
    log_lik[i] = beta_lpdf(y10[i] | mu10[i] * phi10, (1 - mu10[i]) * phi10);
  }
}
