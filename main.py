import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def Bayesian_polynom_regression(x:tf.Tensor, 
                       y:tf.Tensor, 
                       degree:int, 
                       noise_sigma:float=5., 
                       num_steps:int=5000,
                       num_bunrin_steps:int=1000,
                       tune:int=1000,
                       plot_posterior:bool=True,
                       plot_best_fit:bool=True) -> dict:

  """
  Probabilistica Bayessian simple linearn (polynomial of degree 1) and polynomial regression implimenting MCMC (Non-U-turn sampler) for parameter estimation 
  and elpd waic estimation of the model.

  Args:

  x: (1-rank tf.Tensor) predictor,
  y: (1-rank tf.Tensor) predicted,
  degree: (int) degree of the model starting from 1,
  noise_sigma (float): pior standard deviation parameter for noise in data, default = 5.,
  num_steps (int): num steps for MCMC sampler (Non-U-turn), default=5000,
  num_bunrin_steps (int): num burnint steps for MCMC sampler, defaul=1000,
  tune (int): number of initial MCMC samples to be excluded from the final result, default=1000,
  plot_posterior (bool): plot traces and kde of estimated parameters, default=True,
  plot_best_fit (bool): plot resulting fit, default=True

  Returns:

  dict: estimated samples of each of the regression parameters, mean (maximum a posterior) values of each parameter, elpd waic and p waic of the model.

  """
  
  loc_fn = lambda beta: tf.reshape(tf.linalg.matmul(
                        tf.reshape(beta, [1, -1]), 
                        tf.cast(tf.stack([x**pow for pow in range(degree+1)], axis=0), tf.float32)), [-1])

  model = tfd.JointDistributionSequential([tfd.Independent(tfd.Normal(loc=[tf.math.reduce_mean(y)]+ [0.]*degree, 
                                                                    scale=1.), 
                                                         reinterpreted_batch_ndims=1, 
                                                         name='beta'),
                                         tfd.HalfCauchy(loc=0., scale=noise_sigma, name='eps'),
                                         lambda eps, beta: tfd.Independent(tfd.Normal(loc=loc_fn(beta), scale=eps), 
                                                                           reinterpreted_batch_ndims=1)
                                      
                                          ])


  beta_init = tf.cast(tf.fill([int(degree+1),], 1.), tf.float32)
  eps_init = tf.constant(1.)
  state_init = [beta_init, eps_init]

  target_log_prob_fn = lambda *args: model.log_prob(*args, y)

  kernel=tfp.mcmc.NoUTurnSampler(target_log_prob_fn,
                                         step_size=.5)
  transformed_kernel = tfp.mcmc.TransformedTransitionKernel(inner_kernel=kernel, bijector=[tfb.Identity(), tfb.Identity()])
  @tf.function(autograph=False, experimental_compile=True)
  def sample(num_steps=num_steps, num_burnin_steps=num_bunrin_steps):
    adapted_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=transformed_kernel, 
        num_adaptation_steps=num_burnin_steps,
        target_accept_prob=0.9)
    
    trace = tfp.mcmc.sample_chain(num_results=num_steps,
                                  current_state=state_init,
                                  kernel=adapted_kernel,
                                  trace_fn=None)
    return trace



  trace = sample()
  beta = trace[0][tune:, :]
  eps = trace[1][tune:]
  @tf.function(autograph=False, experimental_compile=True)
  def elpd_waic(beta, eps):
    loc_maxtix_fn = lambda params, x: tf.matmul(params, tf.stack([x**pow for pow in range(beta.shape[1])]))
    loc_matrix = loc_maxtix_fn(beta, x)
    log_likelihood_matrix = tfd.Normal(loc=loc_matrix, scale=tf.reshape(eps, [-1, 1])).log_prob(tf.reshape(y, [1,-1]))

    avrg_prob_over_samples = tf.math.reduce_sum(tf.math.exp(log_likelihood_matrix), axis=0)/int(num_steps-tune)
    avrg_log_prob_over_samples = tf.math.log(avrg_prob_over_samples)
    lpd_waic = tf.math.reduce_sum(avrg_log_prob_over_samples, axis=0)

    variance_log_prob_over_samples = tfp.stats.variance(log_likelihood_matrix, sample_axis=0)
    p_waic = tf.math.reduce_sum(variance_log_prob_over_samples, axis=0)
    return dict(elpd_waic=lpd_waic + p_waic, p_waic=p_waic)


  waic = elpd_waic(beta, eps)

  if plot_posterior:

    count = 0
    fig, ax = plt.subplots(beta.shape[1]+1, 2, figsize=(12, 15))
    for var, var_name, idx in zip(tf.transpose(tf.concat([beta, tf.reshape(eps, [-1, 1])], axis=1)), [f'beta{ix}' for ix in range(beta.shape[1])]+['eps'], range(beta.shape[1]+1)):
      ax[idx, count].plot(var)
      ax[idx, count].set_title(var_name)
      count = 1
      sns.kdeplot(var, ax=ax[idx, count])
      ax[idx, count].set_title(var_name)
      count = 0
    plt.tight_layout()
    plt.show()
    
    if plot_best_fit:

      x_val = tf.convert_to_tensor(np.linspace(tf.math.reduce_min(x), tf.math.reduce_max(x), 200), tf.float32)
      y_pred=tf.matmul(tf.cast(tf.reshape(tf.math.reduce_mean(beta, axis=0), [1,-1]), tf.float32), tf.stack([x_val**pow for pow in range(beta.shape[1])]))
      y_pred = tf.reshape(y_pred, [-1])
      plt.scatter(x, y, label='observed')
      plt.plot(x_val, y_pred, label='predicted')
      plt.legend()
      plt.show()
 
  return dict(regression_parameters=beta, 
              mean_parameters_values=tf.math.reduce_mean(beta, axis=0),
              waic=waic)
