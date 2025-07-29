data {
  int<lower = 1> N;                     
  vector<lower = 0, upper = 1>[N] x8;      // year 8 grade
  vector<lower = 0, upper = 1>[N] x9;      // MTH1W grade
  vector<lower = 0, upper = 1>[N] y10;     // MPM2D grade
}

transformed data {
  real mean_x8 = mean(x8);
  real mean_x9 = mean(x9);
}

parameters {
  // year 8
  real<lower = 0, upper = 1> mu8;

  // model for x9
  real alpha9;
  real<lower = 0> beta1;

  // model for y10
  real alpha10;
  real<lower = 0> beta2;
  real<lower = 0> beta3;
  
  // precisions
  real<lower = 0> phi8;                 
  real<lower = 0> phi9;                 
  real<lower = 0> phi10;  
}

transformed parameters {
  vector<lower = 0, upper = 1>[N] mu9;
  vector<lower = 0, upper = 1>[N] mu10;

  for (i in 1:N) {
    mu9[i]  = inv_logit(alpha9 + beta1 * (x8[i] - mean_x8));

    mu10[i] = inv_logit(alpha10 + beta2 * (x8[i] - mean_x8) + beta3 * (x9[i] - mean_x9));
  }
}

model {
  // Priors
  mu8      ~ normal(0.8, 0.1);
  alpha9   ~ normal(1.2, 0.5);
  beta1    ~ normal(5, 1.5);
  alpha10  ~ normal(0.95, 0.5);
  beta2    ~ normal(2, 1);
  beta3    ~ normal(4, 2);
  phi8     ~ normal(10, 3);
  phi9     ~ normal(10, 3);
  phi10    ~ normal(10, 3);

  // Likelihood
  x8   ~ beta(mu8 * phi8, (1 - mu8) * phi8);
  x9   ~ beta(mu9 .* phi9, (1 - mu9) .* phi9);
  y10  ~ beta(mu10 .* phi10, (1 - mu10) .* phi10);
}

generated quantities {
  real indirect_effect;
  real direct_effect;
  real total_effect;

  vector[N] x9_hat;
  vector[N] y10_hat;

  // 1) Mediation effects
  indirect_effect = beta1 * beta3;
  direct_effect   = beta2;
  total_effect    = beta2 + beta1 * beta3;

  // 2) Posterior-predictive draws
  for (i in 1:N) {
    x9_hat[i]  = beta_rng(mu9[i] * phi9, (1 - mu9[i]) * phi9);
    y10_hat[i] = beta_rng(mu10[i] * phi10, (1 - mu10[i]) * phi10);
  }
  
  vector[N] log_lik;

  for (i in 1:N) {
    log_lik[i] = beta_lpdf(y10[i] | mu10[i] * phi10, (1 - mu10[i]) * phi10);
  }
}
