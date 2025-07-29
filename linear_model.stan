data {
  int<lower = 1> N;                     
  vector<lower = 0, upper = 1>[N] x8;      
  vector<lower = 0, upper = 1>[N] x9;      
  vector<lower = 0, upper = 1>[N] y10;     
}

transformed data {
  real mean_x8 = mean(x8);
  real mean_x9 = mean(x9);
}

parameters {
  real<lower = 0, upper = 1> mu8;
  
  real<lower = 0, upper = 1> alpha9;
  real<lower = 0> beta1;

  real<lower = 0, upper = 1> alpha10;
  real<lower = 0> beta2;
  real<lower = 0> beta3;
  
  real<lower = 0> phi8;                 
  real<lower = 0> phi9;                 
  real<lower = 0> phi10;  
}

transformed parameters {
  vector<lower = 0, upper = 1>[N] mu9;
  vector<lower = 0, upper = 1>[N] mu10;

  for (i in 1:N) {
    mu9[i]  = alpha9 
            + beta1  * (x8[i]  - mean_x8);

    mu10[i] = alpha10 
            + beta2  * (x8[i]  - mean_x8)
            + beta3  * (x9[i]  - mean_x9);
  }
}

model {
  // -- Priors --
  mu8 ~ normal(0.8, 0.1);
  
  alpha9 ~ normal(0.77, 0.1);
  beta1 ~ normal(1, 0.5);
  
  alpha10 ~ normal(0.72, 0.1);
  beta2 ~ normal(0.5, 0.5);
  beta3 ~ normal(1, 0.5);
  
  phi8 ~ normal(10, 3);
  phi9 ~ normal(10, 3);
  phi10 ~ normal(10, 3);

  // -- Likelihood --
  x8   ~ beta( mu8 * phi8,    (1 - mu8) * phi8 );
  x9   ~ beta( mu9 * phi9,    (1 - mu9)   * phi9 );
  y10  ~ beta( mu10 * phi10,  (1 - mu10)  * phi10 );
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

  // 2) Posterior‐predictive draws
  for (i in 1:N) {
    x9_hat[i]  = beta_rng( mu9[i]  * phi9,
                           (1 - mu9[i])  * phi9 );
    y10_hat[i] = beta_rng( mu10[i] * phi10,
                           (1 - mu10[i]) * phi10 );
  }
  
  vector[N] log_lik;

  for (i in 1:N) {
    log_lik[i] = beta_lpdf(y10[i] | mu10[i] * phi10, (1 - mu10[i]) * phi10);
    }

}
