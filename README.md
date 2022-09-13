# Bayesian-linear-and-plynomial-regression-and-waic-calculation
Bayesian simple polynomial and linear regression and model estimation via expected log pointwise predictive density (elpd) of widely applicable information criterion (WAIC). WAIC calculation is performed acording to: https://arxiv.org/abs/1507.04544

The file main.py contatins the python function defined as following:

```
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
```



Example of use:

```
#load set 2 from Anscombe's quartet and center x values
import pandas as pd
ans = pd.read_csv('/content/drive/MyDrive/Bayesian Analysis with Python/anscombe.csv')
x_2 = ans.x[ans.group=='II'].values
x_2 = x_2-x_2.mean()
y_2 = ans.y[ans.group=='II'].values
y_2_ans = tf.constant(y_2, tf.float32)
x_2_ans = tf.constant(x_2, tf.float32)
plt.scatter(x_2_ans, y_2_ans)
plt.show()
```

![image](https://user-images.githubusercontent.com/93482551/189919332-4bdd2353-43a0-4fee-93a7-9c22bbc66134.png)

```
#get results of the the regression analysis
result = Bayesian_polynom_regression(x_2_ans, 
                                     y_2_ans, degree=5, 
                                     num_steps=5000, 
                                     num_bunrin_steps=1000, 
                                     tune=1000, plot_posterior=True, 
                                     plot_best_fit=True)
```
![image](https://user-images.githubusercontent.com/93482551/189920002-e53b2c69-0416-4e66-b6c4-110350b43303.png)
![image](https://user-images.githubusercontent.com/93482551/189920215-f9c008ec-682a-45fe-bf55-33bc933eba58.png)

```
result
```
```
{'regression_parameters': <tf.Tensor: shape=(4000, 6), dtype=float32, numpy=
 array([[ 8.7673416e+00,  4.9979320e-01, -1.2687527e-01,  6.5057116e-06,
          6.6257094e-06,  2.6942362e-07],
        [ 8.7673416e+00,  4.9979702e-01, -1.2687472e-01,  6.9389330e-06,
          3.3985787e-06,  3.5009052e-07],
        [ 8.7673416e+00,  4.9990091e-01, -1.2675487e-01,  5.5325712e-05,
         -8.5338058e-07, -1.4824584e-06],
        ...,
        [ 8.7692432e+00,  5.0080472e-01, -1.2694740e-01, -1.4527123e-04,
          7.0962055e-06,  4.1403996e-06],
        [ 8.7686205e+00,  5.0105631e-01, -1.2687364e-01, -2.0972791e-04,
          7.3701517e-06,  7.5047606e-06],
        [ 8.7682304e+00,  5.0119448e-01, -1.2658577e-01, -8.8778383e-05,
         -5.8781488e-06,  1.3778143e-07]], dtype=float32)>,
 'mean_parameters_values': <tf.Tensor: shape=(6,), dtype=float32, numpy=
 array([ 8.7679424e+00,  4.9988589e-01, -1.2659830e-01,  1.3751058e-05,
        -5.1483516e-06, -3.5185911e-07], dtype=float32)>,
 'waic': {'elpd_waic': <tf.Tensor: shape=(), dtype=float32, numpy=56.543797>,
  'p_waic': <tf.Tensor: shape=(), dtype=float32, numpy=4.2208443>}}
  ```
