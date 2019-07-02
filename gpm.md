---
layout: page
permalink: /gpm/
---


I'm guessing that most people are pretty comfortable with the concept of uncorrelated Gaussian noise. It's the most frequently assumed noise. Even if you don't realise it, you're probably assuming Gaussian noise.

Quick check: Are you using a chi-squared test to fit your data? Yes? Well there you go.

- [Co-variate Gaussian Noise](#covariatenoise)
  - [Covariate Gaussian Noise in Python](#covarpython)
- [Gaussian Process Modelling](#gpm)
  - [Gaussian Process Modelling in Python](#gpmpython)
- [Pros/Cons of Nearest Neighbor](#procon)
- [Summary](#summary)
- [Summary: Applying kNN in practice](#summaryapply)
- [Further Reading](#reading)

<a name='covariatenoise'></a>

### Co-variate Gaussian Noise

Here I'm going to talk about **multi-variate**, or **co-variate**, Gaussian noise. Co-variate Gaussian noise is the situation where *the value of one data point affects the value of another*. This kind of *co-*variance, i.e. variance *between* things, is usually expressed as a **covariance matrix**.

If you have **N** data points, then your covariance matrix will have a size: **N x N**. The matrix is normally denoted **K** (or sometimes $latex \mathbf{\Sigma}$).
<p style="text-align:center;">$latex \mathbf{K(x,x)} = \left( \begin{array}{cccc} k(x_1,x_1) & k(x_1,x_2) & ... & k(x_1,x_n) \\ k(x_2,x_1) & k(x_2,x_2) & ... & k(x_2,x_n) \\ \vdots & \vdots & \vdots & \vdots \\ k(x_n,x_1) & k(x_n,x_2) & ... & k(x_n,x_n)  \end{array} \right) $</p>

<strong>Uncorrelated</strong>, or <em>independent</em>, Gaussian noise is a special case of the covariance matrix where only the diagonal elements have a non-zero value, i.e.
<p style="text-align:center;">$latex \mathbf{K(x,x)} = \left( \begin{array}{cccc} k(x_1,x_1) & 0 & ... & 0 \\ 0 & k(x_2,x_2) & ... & 0 \\ \vdots & \vdots & \vdots & \vdots \\ 0 & 0 & ... & k(x_n,x_n)  \end{array} \right) $</p>

The value of each diagonal element corresponds to the variance of a particular data point, e.g. a data point at position $latex x_1$ with a mean value $latex \mu_1 $ would have a variance $latex k(x_1,x_1) = \sigma_1^2 $ ; a data point at position $latex x_2$ with a mean value $latex \mu_2 $ would have a variance $latex k(x_2,x_2) = \sigma_2^2 $ , and so on and so forth.

The actual <em>y</em>-value at each <em>x</em>-position will be drawn from a probability distribution
<p style="text-align:center;">$latex p(y_i) = N (\mu_i, K_{ij})$</p>
which in the case of a diagonal covariance matrix reduces to
<p style="text-align:center;">$latex y_i = \mu_i + N(0, \sigma^2_i)$.</p>
However, if we start to add in non-zero values to the other elements of the covariance matrix then this will no longer be the case and the <em>y</em>-value at one position will affect the <em>y</em>-value at another.

<a name='reading'></a>

<h4>Reading Material</h4>

This is a very brief explanation of covariate Gaussian noise. For a better and more detailed description, I like these references:
<ul>
	<li><a href="http://www.gaussianprocess.org" target="_blank" rel="noopener">Gaussian Processes for Machine Learning</a>, Carl Edward Rasmussen and Chris Williams, the MIT Press</li>
	<li><a href="http://www.robots.ox.ac.uk/~sjrob/Pubs/philTransA_2012.pdf" target="_blank" rel="noopener">Gaussian processes for time-series modelling</a>, S. Roberts, M. Osborne, M. Ebden, S. Reece, N. Gibson and S. Aigrain, Phil. Trans. R. Soc. A 2013 371, 20110550</li>
</ul>
The second of these has a particularly nice figure showing the effect of covariate Gaussian noise. It looks like this:

<div class="fig figcenter fighighlight">
  <img src="https://allofyourbases.files.wordpress.com/2017/08/roberts_fig5.png">
  <div class="figcaption">Figure 5 from Roberts et al. 2012.</div>
</div>

Here I'm going to explain how to recreate this figure using Python.

<a name='covarpython'></a>

<h4>Covariate Gaussian Noise in Python</h4>

To simulate the effect of co-variate Gaussian noise in Python we can use the <a href="http://www.numpy.org/" target="_blank" rel="noopener">numpy</a> library function <code>multivariate_normal(mean,K)</code>.

<em>Note: the Normal distribution and the Gaussian distribution are the same thing.</em>

First off, let's load some libraries:

```python
import numpy as np   # the numpy library
import pylab as pl   # the matplotlib for plotting
```

then we can call the <code>multivariate_normal(mean,K)</code> function:

```python
# make an array of positions
# these are evenly spaced, but they don’t have to be
x = np.arange(0, 20.,0.01)

# use numpy to draw a sample from the multi-variate
# normal distribution, N(0,K), at all the positions in x
y = np.random.multivariate_normal(np.zeros(len(x)),K)
```

The <code>multivariate_normal</code> function takes two arguments: (1) an array of noiseless mean values for each of the x-positions, and (2) a covariance matrix for all the x-positions.

In this case, I've made all of the  mean values equal to zero.

At the moment we haven't specified <strong>K</strong>, so these lines of code won't work just yet. To create <strong>K</strong> we need to build a matrix of values that are calculated by the function little <strong>k</strong>. This function is known as the <strong>covariance kernel</strong> and it defines how much of an affect one data value has on another.

<h3>The Covariance Kernel</h3>

To start with we are going to define a <em>squared-exponential covariance kernel</em>. This has the form:
<p style="text-align:center;">$latex k(x_n,x_m) = h^2 \exp{ \left( \frac{-(x_n - x_m)^2}{\lambda^2} \right)}
$</p>

where $latex x_n$ is the x-position of one data point and $latex x_m $ is the x-position of another. The value of the kernel is a function of how far away from each other these data points are, i.e. $latex x_n - x_m $. It also has a couple of <em>hyper-</em>parameters that govern the overall shape of the kernel: $latex h$ and $latex \lambda$. These are referred to as <strong>hyper-parameters</strong> because they don't really have any direct physical meaning so they're not bog-standard normal model parameters. Here the $latex h $ parameter controls the <em>normalisation</em> of the kernel and $latex \lambda $ controls the <em>width</em> of the kernel.

```python
def cov_kernel(x1,x2,h,lam):

    """
    Squared-Exponential covariance kernel
    """

    k12 = h**2*np.exp(-1.*(x1 - x2)**2/lam**2)
    return k12
```

We can then use this kernel function to fill our covariance matrix:

```python
def make_K(x, h, lam):

    """
    Make covariance matrix from covariance kernel
    """

    # for a data array of length x, make a covariance matrix x*x:
    K = np.zeros((len(x),len(x)))

    for i in range(0,len(x)):
        for j in range(0,len(x)):

            # calculate value of K for each separation:
            K[i,j] = cov_kernel(x[i],x[j],h,lam)

     return K
```

and we can then update our earlier call to the numpy <code>multivariate_normal</code> function:

```python
# make an array of 200 evenly spaced positions between 0 and 20:
x = np.arange(0, 20.,0.01) 

# make a covariance matrix:
K = make_K(x,h,lam)

# draw samples from a co-variate Gaussian
# distribution, N(0,K), at positions x1:
y = np.random.multivariate_normal(np.zeros(len(x)),K)
```

<h3>Putting It All Together</h3>
We can expand this to look at what happens when we vary the hyper-parameters of the covariance kernel, in particular the width, $latex \lambda $:

```python
# make an array of 200 evenly spaced positions between 0 and 20:
x = np.arange(0, 20.,0.01)

for i in range(0,3):

    h = 1.0

    if (i==0): lam = 0.1
    if (i==1): lam = 1.0
    if (i==2): lam = 5.0

    # make a covariance matrix:
    K = make_K(x,h,lam)

    # five realisations:
    for j in range(0,5):

        # draw samples from a co-variate Gaussian distribution, N(0,K):
        y = np.random.multivariate_normal(np.zeros(len(x)),K)

        tmp2 = '23'+str(i+3+1)
        pl.subplot(int(tmp2))
        pl.plot(x,y)

    tmp1 = '23'+str(i+1)
    pl.subplot(int(tmp1))
    pl.imshow(K)
    pl.title(r"$\lambda = $"+str(lam))

pl.show()
```

<div class="fig figcenter fighighlight">
  <img src="https://allofyourbases.files.wordpress.com/2017/08/figure_1.png">
  <div class="figcaption"></div>
</div>


The top row of plots is showing the covariance matrix for each of the three different kernel widths. The bottom row is showing five realisations of data <em>y</em>-values that correspond to each of the covariance matrices. Remember that all of these realisations have <strong>zero mean</strong> so the only contribution to the <em>y</em>-values is (covariate) Gaussian noise.

The narrower the covariance kernel, the more diagonal the covariance matrix becomes and the more like independent noise the <em>y</em>-values appear.

<a name='gpm'></a>

<h3>Gaussian Process Modelling</h3>

We can invert the probability statement
<p style="text-align:center;">$latex p(y_i) = N (\mu_i, K_{ij})$</p>
to calculate the value of a hypothetical measurement at a test position, $latex x_{\ast}$, based on existing measurements, <em>y</em>, at positions, <em>x</em>.

Key to this are two things:
<ol>
	<li>Knowing the form of the covariance <em>a priori</em>; or, having sufficient example data points to calculate it;</li>
	<li>Having a sufficient number of example data points to provide fixed points in the function <em>a priori</em>. These are our <strong>training data</strong>.</li>
</ol>
Normally if you can satisfy the second of these requirements, the first follows naturally.

Inverting the above probability statement allows us to write the following equations. These give us the <strong>posterior mean</strong> ($latex m_{\ast}$) and the <strong>posterior variance </strong>($latex C_{\ast}$) at that point, i.e. <strong>the value</strong> and <strong>the uncertainty on that value</strong>:
<p style="text-align:center;">$latex
\mathbf{m_{\ast}} = \mathbf{ K(x_{\ast},x)^T K(x,x)^{-1} y } \\
\mathbf{C_{\ast}} = \mathbf{ K(x_{\ast},x_{\ast}) - K(x_{\ast},x)^T K(x,x)^{-1} K(x_{\ast},x) }
$</p>
This is more simply written as:
<p style="text-align:center;">$latex
\mathbf{m_{\ast}} = \mathbf{ k^T_{\ast} K^{-1} y } \\
\mathbf{C_{\ast}} = \mathbf{ k(x_{\ast},x_{\ast}) - k^T_{\ast} K^{-1} k_{\ast} }
$</p>
assuming that the original measurements have zero mean. If they don't it becomes:
<p style="text-align:center;">$latex
\mathbf{m_{\ast}} = \mu_{\ast} + \mathbf{ k^T_{\ast} K^{-1} (y - \mu) } \\
\mathbf{C_{\ast}} = \mathbf{ k(x_{\ast},x_{\ast}) - k^T_{\ast} K^{-1} k_{\ast} }.
$</p>
To understand where this comes from in more detail you can read <a href="http://www.gaussianprocess.org/gpml/chapters/RW2.pdf" target="_blank" rel="noopener">Chapter 2 of Rasmussen & Williams</a>.

So, how do we actually practically do this in Python?

<a name='gpmpython'></a>

<h3>Prediction in Python</h3>

In <a href="http://allofyourbases.com/2017/08/21/gaussian-processes-in-python/" target="_blank" rel="noopener">a previous post</a> I went through the steps for simulating covariate data in Python.

We ended up with some covariate data that looked like this:

<img class="  wp-image-1878 aligncenter" src="https://allofyourbases.files.wordpress.com/2017/08/figure_1.png" alt="figure_1" width="502" height="385" />

If we take the final realization from these data, which has $latex \lambda = 5$, and select 5 points from it as our <strong>training data</strong> then we can calculate <strong>the posterior mean and variance</strong> at any other point based on those five training points.
<p style="text-align:center;">[<em>Note: This code continues directly on from the code <a href="http://allofyourbases.com/2017/08/21/gaussian-processes-in-python/" target="_blank" rel="noopener">in the previous post</a></em>]</p>
First off, let's randomly select our training points and allocate all the data positions in our realisation as either <em>training</em> or <em>test</em>:

```python
# set number of training points
nx_training = 5

# randomly select the training points:
tmp = np.random.uniform(low=0.0, high=2000.0, size=nx_training)
tmp = tmp.astype(int)

condition = np.zeros_like(x)
for i in tmp: condition[i] = 1.0

y_train = y[np.where(condition==1.0)]
x_train = x[np.where(condition==1.0)]
y_test = y[np.where(condition==0.0)]
x_test = x[np.where(condition==0.0)]
```

Next we can use those 5 training data points to make a <strong>covariance matrix</strong> using the function we defined <a href="http://allofyourbases.com/2017/08/21/gaussian-processes-in-python/" target="_blank" rel="noopener">in the previous post</a>:

```python

# define the covariance matrix:
K = make_K(x_train,h,lam)

```

To calculate the posterior mean and variance we're going to need to calculate the <strong>inverse</strong> of our covariance matrix. For a small matrix like we have here, we can do this using the numpy library linear algebra functionality:

```python

# take the inverse:
iK = np.linalg.inv(K)
```

However, <em>be careful inverting larger matrices in Python</em>. The numerical stability of the numpy linear algebra inversion is pretty poor. The scipy linear algebra inversion is better because it always uses <a href="http://www.netlib.org/blas/" target="_blank" rel="noopener">BLAS</a>/<a href="http://www.netlib.org/lapack/" target="_blank" rel="noopener">LAPACK</a>, but whenever possible <a href="https://stackoverflow.com/questions/8690456/numerically-stable-inverse-of-a-2x2-matrix" target="_blank" rel="noopener">don't invert a matrix at all</a>.

And now we're ready to calculate the posterior mean and variance (or standard deviation, which is the square-root of the variance):

```python

m=[];sig=[]
for xx in x_test:

    # find the 1d covariance matrix:
    K_x = cov_kernel(xx, x_train, h, lam)

    # find the kernel for (xx,xx):
    k_xx = cov_kernel(xx, xx, h, lam)

    # calculate the posterior mean and variance:
    m_xx = np.dot(K_x.T,np.dot(iK,y_train))
    sig_xx = k_xx - np.dot(K_x.T,np.dot(iK,K_x))

    m.append(m_xx)
    sig.append(np.sqrt(np.abs(sig_xx))) # note sqrt to get stdev from variance

```

Let's see how we did. Here I'm going to plot the training data points, as well as the original realisation (dashed line) that we drew them from, on the <strong>left</strong>. On the <strong>right</strong> I'm going to plot the same data plus the predicted posterior mean (solid line) and the standard deviation (shaded area).

```python

# m and sig are currently lists - turn them into numpy arrays:
m=np.array(m);sig=np.array(sig)

# make some plots:

# left-hand plot
ax = pl.subplot(121)
pl.scatter(x_train,y_train)  # plot the training points
pl.plot(x,y,ls=':')        # plot the original data they were drawn from
pl.title("Input")

# right-hand plot
ax = pl.subplot(122)
pl.plot(x_test,m,ls='-')     # plot the predicted values
pl.plot(x_test,y_test,ls=':') # plot the original values

# shade in the area inside a one standard deviation bound:
ax.fill_between(x_test,m-sig,m+sig,facecolor='lightgrey', lw=0, interpolate=True)
pl.title("Predicted")

pl.scatter(x_train,y_train)  # plot the training points

# display the plot:
pl.show()

```

<img class="  wp-image-1976 aligncenter" src="https://allofyourbases.files.wordpress.com/2017/08/gpm1.png" alt="GPM1.png" width="491" height="368" />

We can see that the prediction is pretty good. It's not exactly the same as the original realisation, but then again it would be pretty surprising if we could predict things perfectly based on incomplete data.

The shaded regions of uncertainty are also useful because they tell us how much accuracy we should expect in any given range of positions. Note that the further away from our training data we go, the larger the uncertainty becomes. On the right hand side it explodes because we've run out of training data points to constrain the posterior mean.

Statistically we should not expect our original realisation to lie within the shaded region all the time: it's a one sigma limit, i.e. we should expect the values of the original realisation to lie within one sigma of the posterior mean ~68% of the time.
