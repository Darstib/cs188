---
tags:
  - notes
comments: true
dg-publish: true
---

> [!PREREQUISITE]
>
> - [07-Propositional_Logic_and_Planning](../note/07-Propositional_Logic_and_Planning.md)
> - [08-DPLL&ForwardChaining](../note/08-DPLL&ForwardChaining.md)
> - [09-First_Order_Logic](../note/09-First_Order_Logic.md)
> - [10-Intro_to_Probability](../note/10-Intro_to_Probability.md)
> - [project 3](https://inst.eecs.berkeley.edu/~cs188/sp24/projects/proj3/)（若需要认证，可见[仓库](https://github.com/Darstib/cs188/tree/main/materials/project/intro_page)）

实话说，project 3 相比于考察逻辑运算，个人感觉考察 python 更加多；因为在讲解实验时，文档中给出了很多伪代码；哪怕我们不知道它在干什么，纯靠 python 编程实现它似乎不是什么难事（但是我python 编程确实不是很行 😇）。

同时基于此时其他比较需要花费时间的工作，我更多只是结合 project 3 将 [pavlosdais 的代码](https://github.com/pavlosdais/ai-berkeley/blob/main/Project%203%20-%20Logic/logicPlan.py)自己跑了一遍作罢。该代码基于cs188 sp22 的project 完成，部分函数名和一些些细节与 sp24 的略有不同，但本身讲的比较详细了，仔细读就能找到不同之处。

- 对于Q1-Q5 看的比较仔细，也赞叹于将逻辑表达式“对象化”的操作；
- 对于Q6-Q8 看的比较粗略；虽然后三问大同小异，还是佩服 pavlosdais 能够很好地将实验文档中的伪代码实例化。

我自己略作修改后的代码也会放在 [solution 文件夹](https://github.com/Darstib/cs188/tree/main/project/solution)中，已 [运行全部通过](attachments/project-3.png)。