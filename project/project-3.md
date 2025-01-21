---
tags:
  - notes
data: 2024-08-11
comments: true
dg-publish: true
---

# Logic

> [!PREREQUISITE]
>
> - [07-Propositional_Logic_and_Planning](../note/07-Propositional_Logic_and_Planning.md)
> - [08-DPLL&ForwardChaining](../note/08-DPLL&ForwardChaining.md)
> - [09-First_Order_Logic](../note/09-First_Order_Logic.md)
> - [10-Intro_to_Probability](../note/10-Intro_to_Probability.md)
> - [project 3](https://inst.eecs.berkeley.edu/~cs188/sp24/projects/proj3/)（若需要认证，可见[仓库](https://github.com/Darstib/cs188/tree/main/materials/project/intro_page)）

## Quick Review

- DPLL（Davis-Putnam-Logemann-Loveland 算法）
	- DPLL 算法是一种深度优先回溯搜索算法，旨在解决可满足性问题（SAT），即给定一个逻辑句子，找到所有符号的有效赋值。该算法通过三种技巧减少过度回溯：早期终止、纯符号启发式和单元子句启发式。DPLL 处理的输入是合取范式（CNF），并通过不断赋值符号的真值，直到找到满足模型或无法赋值为止。
- Forward Chaining（前向推理）
    - 前向推理是一种推理方法，常用于解决可满足性问题（SAT）。它通过从已知的事实出发，逐步应用规则来推导新的事实，直到达到目标或无法再推导出新事实为止。在SAT中，前向推理可以用于从已知的逻辑句子中推导出新的赋值，从而帮助确定是否存在满足条件的模型。
- First-order Logic（第一阶逻辑，FOL）
	- 第一阶逻辑是一种形式逻辑系统，使用量词和变量来表达关于对象的命题。与命题逻辑不同，第一阶逻辑允许使用量词（如“对于所有”和“存在”）来处理更复杂的逻辑关系。

## explain

实话说，project 3 相比于考察逻辑运算，个人感觉考察 python 更加多；因为在讲解实验时，文档中给出了很多伪代码；哪怕我们不知道它在干什么，纯靠 python 编程实现它似乎不是什么难事。

同时基于此时其他比较需要花费时间的工作，我更多只是结合 project 3 将 [pavlosdais 的代码](https://github.com/pavlosdais/ai-berkeley/blob/main/Project%203%20-%20Logic/logicPlan.py)自己跑了一遍作罢。该代码基于cs188 sp22 的project 完成，部分函数名和一些些细节与 sp24 的略有不同，但本身讲的比较详细了，仔细读就能找到不同之处。

- 对于Q1-Q5 看的比较仔细，也赞叹于将逻辑表达式“对象化”的操作；
- 对于Q6-Q8 看的比较粗略；虽然后三问大同小异，还是佩服 pavlosdais 能够很好地将实验文档中的伪代码实例化。

## pass

我自己略作修改后的代码也会放在 [solution 文件夹](https://github.com/Darstib/cs188/tree/main/project/solution)中，已 [运行全部通过](attachments/project-3.png)。