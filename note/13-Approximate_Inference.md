---
tags:
  - notes
  - cs188
comments: true
dg-publish: true
---

An alternate approach for probabilistic reasoning is to implicitly calculate the probabilities for our query by simply counting samples. This will not yield the exact solution, as in IBE or Variable Elimination, but this approximate inference is often good enough, especially when taking into account **massive savings in computation.**

## note

### Prior Sampling

Given a Bayes Net model, we can easily write a simulator. For example, consider the CPTs given below for the simplified model with only two variables T and C. We call this simple approach **prior sampling**. 

![](attachments/13_Approximate%20Inference.png)

```python title="prior sampling"
import random

def get_t():
	if random.random() < 0.99:
	    return True
	return False
	
def get_c(t):
    if t and random.random() < 0.95:
        return True
    return False
    
def get_sample():
    t = get_t()
    c = get_c(t)
    return [t, c]
```

The downside of this approach is that it may require the generation of a very large number of samples in order to perform analysis of unlikely scenarios. If we wanted to compute P(C| −t), we’d have to throw away 99% of our samples.

### Rejection Sampling

One way to mitigate the previously stated problem is to modify our procedure to early reject any sample inconsistent with our evidence. Namly if we want to generate P(C|-t), we don't generate c if T=+t so we can throw most of bad samples to faster generating. 

### Likelihood weighting

A more exotic approach is **likelihood weighting,** which <u>ensures that we never generate a bad sample. In this approach, we manually set all variables equal to the evidence in our query.</u>

For example, if we wanted to compute P(C|−t), we’d simply declare that t is false. The problem here is that this may yield samples that are inconsistent with the correct distribution. If we simply force some variables to be equal to the evidence, then our samples occur with probability only equal to the products of the CPTs of the non-evidence variables. This means the joint PDF has no guarantee of being correct (though may be for some cases like our two variable Bayes Net).

Likelihood weighting solves this issue by using a weight for each sample, which is the probability of the evidence variables given the sampled variables.

```python title="Likelihood weighting"
def likelihood_weighting(X, e, bn, N):
    """
    Calculate the posterior probability P(X | e) for the query variable X given evidence e.
    
    Parameters:
    X: Query variable
    e: Observed values for evidence variables
    bn: Bayesian network specifying the joint distribution P(X1, ..., Xn)
    N: Total number of samples to be generated
    
    Returns:
    Normalized weight vector W
    """
    W = {value: 0 for value in X.values()}  # Initialize weight vector

    for j in range(N):
        x, w = weighted_sample(bn, e)  # Generate sample and weight
        W[x] += w  # Update weight

    return normalize(W)  # Normalize weights

def weighted_sample(bn, e):
    """
    Generate an event and a weight.
    
    Parameters:
    bn: Bayesian network
    e: Observed values for evidence variables
    
    Returns:
    x: Generated event
    w: Weight
    """
    w = 1  # Initialize weight to 1
    x = {}  # Initialize event

    for Xi in bn.variables:  # Iterate over all variables
        if Xi in e:  # If it is an evidence variable
            w *= P(Xi | parents(Xi))  # Update weight
            x[Xi] = e[Xi]  # Set event value to the evidence value
        else:
            x[Xi] = random_sample(P(Xi | parents(Xi)))  # Sample from the conditional distribution

    return x, w  # Return event and weight

def normalize(W):
    """
    Normalize the weight vector.
    
    Parameters:
    W: Weight vector
    
    Returns:
    Normalized weight vector
    """
    total_weight = sum(W.values())
    return {key: value / total_weight for key, value in W.items()}  # Normalize
```

For all three of our sampling methods (prior sampling, rejection sampling, and likelihod weighting), we can get increasing amounts of accuracy by generating additional samples. However, of the three, <u>likelihood weighting is the most computationally efficient</u> , for reasons beyond the scope of this course.

### Gibbs Sampling

**Gibbs Sampling** is a fourth approach for sampling. In this approach, we first set all variables to some totally random value (not taking into account any CPTs). We then repeatedly pick one variable at a time, clear its value, and resample it given the values currently assigned to all other variables.

```python title="Gibbs sampling"
def gibbs_ask(X, e, bn, N):
    """
    Returns an estimate of P(X | e) using Gibbs sampling.
    
    Parameters:
    X: Query variable
    e: Observed values for evidence variables
    bn: Bayesian network
    N: Total number of samples to be generated
    
    Returns:
    Normalized count vector N
    """
    N = {value: 0 for value in X.values()}  # Initialize count vector for each value of X
    Z = get_non_evidence_variables(bn)  # Get non-evidence variables in the Bayesian network
    x = initialize_with_random_values(e, Z)  # Initialize x with random values for variables in Z

    for j in range(N):  # Loop for N samples
        for Zi in Z:  # Iterate over each non-evidence variable
            # Set the value of Zi in x by sampling from P(Zi | mb(Zi))
            x[Zi] = sample_from_conditional_distribution(Zi, x)  
            N[x] += 1  # Increment the count for the current value of X

    return normalize(N)  # Normalize the count vector
```

## link

- [cs188-sp24-note13](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/notes/cs188-sp24-note13.pdf) 
- [A Gentle Introduction to Bayesian Deep Learning](https://towardsdatascience.com/a-gentle-introduction-to-bayesian-deep-learning-d298c7243fd6)