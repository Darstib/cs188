---
tags:
  - notes
  - cs188
comments: true
dg-publish: true
---

## note

We’ll now cover a very intrinsically related structure called a Markov model, which for the purposes of this course can be thought of as analogous to a chain-like, infinite-length Bayes’ net. Now, we'll take a weather prediction example.

![](attachments/14-Markov_Models.png)

$$
P(W_0,W_1,...,W_n)=P(W_0)P(W_1|W_0)P(W_2|W_1)...P(W_n|W_{n-1})=P(W_0)\prod_{i=0}^{n-1}P(W_{i+1}|W_i)
$$

To track how our quantity under consideration (in this case, the weather) changes over time, we need to know both it’s **initial distribution** at time t = 0 and some sort of **transition model** that characterizes the probability of moving from one state to another between timesteps.

### mini-forward algothrim

By properties of marginalization, we know that:

$$
P(W_{i+1})=\sum_{w_i}P(w_i,W_{i+1}) \overset{\text{chain rule}}{\implies} P(W_{i+1})=\sum_{w_i}P(W_{i+1}|w_i)P(w_i)
$$

To compute the distribution of the weather at timestep i+1, we look at the probability distribution at timestep i given by $P(W_i)$ and "advance" this model a timestep with our transition model $P(W_{i+1}|W_i)$.

### Stationary Distribution

A new question: does the probability of being in a state at a given timestep ever converge?

To solve this problem, we must compute the stationary distribution of the weather. As the name suggests, the stationary distribution is one that remains the same after the passage of time, i.e. $P(W_{i+1})=P(W_{i})$.

So we have $P(W_{t})=\sum_{w_{t}}P(W_{t+1}|w_{t})P(w_{t})$, in our weather forecast, that is:

$$
\begin{cases}
P(W_t=sun)=P(W_{t+1}=sun|W_t=sun)P(W_t=sun)+P(W_{t+1}=sun|W_t=rain)P(W_t=rain)  \\
P(W_t=rain)=P(W_{t+1}=rain|W_t=sun)P(W_t=sun)+P(W_{t+1}=rain|W_t=rain)P(W_t=rain)
\end{cases}
$$

Since we know $P(w_{t+1}|w_{t})$ (namly the transition model), then we can solve binary first order equations and get $P(W_{t})$.

As expected, $P(W_{\infty+1}) = P(W_{\infty})$. In general, if $W_{t}$ had a domain of size k, the equivalence $P(W_{t})=\sum_{w_{t}}P(W_{t+1}|w_{t})P(w_{t})$ yields a system of k equations, which we can use to solve for the stationary distribution.

## link

- [cs188-sp24-note14](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/notes/cs188-sp24-note14.pdf) 