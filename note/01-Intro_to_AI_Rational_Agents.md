---
comments: true
dg-publish: true
tags:
  - notes
---

> [!PREREQUISITE]
>
> - [有限状态机](https://www.wikiwand.com/zh/articles/%E6%9C%89%E9%99%90%E7%8A%B6%E6%80%81%E6%9C%BA)

## note

### Agents

- rational agent
    - an entity that has goals or preferences and tries to perform a series of actions that yield the best/optimal expected outcome given these goals.
- reflex agent
    - one that doesn’t think about the consequences of its actions, but rather selects an action based solely on the current state of the world.
- PEAS (Performance Measure, Environment, Actuators, Sensors)

<u>The design of an agent heavily depends on the type of environment the agents acts upon</u> . We can characterize the types of environments in the following ways:

- In **partially observable environments（部分可观测环境）**, the agent does not have full information about the state and thus the agent must have an internal estimate of the state of the world. This is in contrast to fully observable environments, where the agent has full information about their state.
- **Stochastic environments（随机环境）** have uncertainty in the transition model, i.e. taking an action in a specific state may have multiple possible outcomes with different probabilities. This is in contrast to **deterministic environments（确定环境）**, where taking an action in a state has a single outcome that is guaranteed to happen.
- In **multi-agent environments（多智能体环境）** the agent acts in the environments along with other agents. For this reason the agent might need to randomize its actions in order to avoid being “predictable" by other agents.
- If the environment does not change as the agent acts on it, then this **environment is called static**. This is contrast to **dynamic environments** that change as the agent interacts with it.
- If an environment has **known physics**, then the transition model (even if stochastic) is known to the agent and it can use that when planning a path. If the **physics are unknown** the agent will need to take actions deliberately to learn the unknown dynamics.

> [!HELP]
>
> 在这里，`know physics` 应该是指已经被人类探索出来的规律；而 `unknown physics` 则是指尚且没有得知的规律。

## link

- [cs188-sp24-note01](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/notes/cs188-sp24-note01.pdf)
