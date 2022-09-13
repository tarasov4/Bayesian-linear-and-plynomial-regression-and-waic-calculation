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
One can run estimation of several polynomial models with inreasing degree starting from 1 (linear model) than choose the best model on the basis of elpd WAIC value:

```
elpd_waics = []
for i in range(1, 8):
  result = Bayesian_polynom_regression(x_2_ans, 
                                          y_2_ans, 
                                          degree=i, 
                                          num_steps=5000, 
                                          num_bunrin_steps=1000, 
                                          tune=1000, 
                                          plot_posterior=False, 
                                          plot_best_fit=True)
  elpd_waics.append(result['waic']['elpd_waic'])


plt.plot(range(1, 8), elpd_waics) 
```
![image](https://user-images.githubusercontent.com/93482551/189928332-02335976-5864-4aa8-ac05-0933e4dd143c.png)
![image](https://user-images.githubusercontent.com/93482551/189929009-69dc8b3f-4e47-4e6b-9e2e-4d92cbe43c03.png)
![image](https://user-images.githubusercontent.com/93482551/189929042-250a3165-2479-4d53-88da-b07a0841c730.png)
![image](https://user-images.githubusercontent.com/93482551/189929067-3d3e8081-f574-4a51-822a-d62fcb6457a7.png)
![image](https://user-images.githubusercontent.com/93482551/189932146-9e9fc168-5162-4066-a99c-a066db050122.png)
![image](https://user-images.githubusercontent.com/93482551/189929133-f9f8c503-0717-42d3-b112-f2824d00cb34.png)
![image](https://user-images.githubusercontent.com/93482551/189929148-76b9c35a-9740-4637-b2b7-0c93c1abbf36.png)
![image](https://user-images.githubusercontent.com/93482551/189929181-2aa8216e-2eb0-4f10-9fce-b4969f4869c7.png)

```
elpd_waics
```

```
[<tf.Tensor: shape=(), dtype=float32, numpy=-15.294616>,
 <tf.Tensor: shape=(), dtype=float32, numpy=56.897133>,
 <tf.Tensor: shape=(), dtype=float32, numpy=56.882465>,
 <tf.Tensor: shape=(), dtype=float32, numpy=57.133026>,
 <tf.Tensor: shape=(), dtype=float32, numpy=56.17394>,
 <tf.Tensor: shape=(), dtype=float32, numpy=-7.6088023>,
 <tf.Tensor: shape=(), dtype=float32, numpy=-23.876358>]
 ```
 
 So, the polynomial model of degree 4 is the best one for approximation the second group of the Anscombe's quartet.
 
 The loaded libraries:
 ```
 import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
