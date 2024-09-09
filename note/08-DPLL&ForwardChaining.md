---
tags:
  - notes
comments: true
dg-publish: true
---

## note

### Model Checking

The inference problem can be formulated as figuring out whether KB ⊨ q, where KB is our knowledge base of logical sentences, and q is some query.

One simple algorithm for checking whether KB ⊨ q is to enumerate all possible models[^0], and to check if in all the ones in which KB is true, q is true as well (using truth table). This approach is known as **model checking**.

[^0]: model 可以理解为对原子命题的真假性进行假定；例如 A∨¬B，那么 {A:T, B:F} 就是一个 model，此时的 q 也确实为 true。 

For a propositional logical system, if there are N symbols, there are $2^N$ models to check, and hence the time complexity of this algorithm is $O(2^N)$, while in first-order logic[^1], the number of models is infinite. 

[^1]: [**First-order logic**](https://www.wikiwand.com/en/articles/First-order_logic) —also called **predicate logic**, **predicate calculus**, **quantificational logic**. Propositional Logic 处理简单命题及其连接；而 Predicate Logic 处理带有量词和变量的命题，那么需要考虑的情况变得无比地多；我们将在下一篇笔记详解讲解。

> [!INFO]
>
> In fact the problem of propositional entailment is known to be co-NP-complete. While the worst case runtime will inevitably be an exponential function of the size of the problem, there are algorithms that can in practice terminate much more quickly. We will discuss two model checking algorithms for propositional logic.

The first, proposed by Davis, Putnam, Logemann, and Loveland (which we will call the **DPLL algorithm**) is essentially a **depth-first, backtracking search** over possible models with three tricks to reduce excessive backtracking. <u>This algorithm aims to solve the satisfiability problem, i.e. given a sentence, find a working assignment to all the symbols.</u> 

As we mentioned, the problem of entailment can be reduced to one of satisfiability[^2] (show that A∧ ¬B is not satisfiable), and specifically <u>DPLL takes in a problem in CNF</u> . Satisfiability can be formulated as a constraint satisfaction problem as follows: let the variables (nodes) be the symbols and the constraints be the logical constraints imposed by the CNF. Then DPLL will continue assigning symbols truth values until either a satisfying model is found or a symbol cannot be assigned without violating a logical constraint, at which point the algorithm will backtrack to the last working assignment

[^2]: **Entailment to Satisfiability**: To check if ( A ) entails ( B ) (written as ( $A \models B$ )), you can instead check if ( $A \land \neg B$ ) is unsatisfiable. If ( $A \land \neg B$ ) cannot be true, then ( $A \models B$ ) is true.

However, DPLL makes three improvements over simple backtracking search:

1. **Early Termination**: A clause is true if any of the symbols are true. Also, a sentence is false if any single clause is false.
2. **Pure Symbol Heuristic**: A pure symbol is a symbol that only shows up in its positive form (or only in its negative form) throughout the entire sentence. Pure symbols can immediately be assigned true or false.[^3]
3. **Unit Clause Heuristic**: A unit clause is a clause with just one literal or a disjunction with one literal and many falses. In a unit clause, we can immediately assign a value to the literal, since there is only one valid assignment.[^4]

[^3]: For example, in the sentence (A∨B)∧(¬B∨C)∧(¬C∨A), we can identify A as the only pure symbol and can immediately A assign to true, reducing the satisfying problem to one of just finding a satisfying assignment of (¬B∨C).
[^4]: For example, B must be true for the unit clause (B∨ false∨ ··· ∨ false) to be true.

### DPLL: Example

Suppose we have the following sentence in conjunctive normal form (CNF), We want to use the DPLL algorithm to determine whether it is satisfiable.

`(¬N ∨ ¬S)∧(M ∨Q∨N)∧(L∨ ¬M)∧(L∨ ¬Q)∧(¬L∨ ¬P)∧(R∨P∨N)∧(¬R∨ ¬L)∧(S)`

> Suppose we use a fixed variable ordering (alphabetical order) and a fixed value ordering (true before false).

On each recursive call to the DPLL function, we keep track of three things:

- _model_ is a list of the symbols we’ve assigned so far, and their values.[^6]
- _symbols_ is a list of unassigned symbols that still need assignments.
- _clauses_ is a list of clauses (disjunctions) in CNF that still need to be considered on this call or future recursive calls to DPLL.

[^6]: For example, {A:T, B:F} tells us the values of two symbols assigned so far.

We start by calling DPLL with an empty model (no symbols assigned yet), symbols containing all the symbols in the original sentence, and clauses containing all the clauses in the original sentence.

```python title="pseudocode1 for DPLL"
function DPLL-SATISFIABLE?(s) returns true or false  
inputs: s, a sentence in propositional logic  
   # 将输入的命题逻辑句子转换为 CNF 形式的子句集合  
   clauses ← the set of clauses in the CNF representation of s  
   # 获取所有的命题符号  
   symbols ← a list of the proposition symbols in s  
   # 调用 DPLL 算法进行求解  
   return DPLL(clauses, symbols, {})
```

> [!EXAMPLE]
>
> - model: {}
> - symbols: [L,M,N,P,Q,R,S]
> - clauses: (¬N∨¬S)∧(M∨Q∨N)∧(L∨¬M)∧(L∨¬Q)∧(¬L∨¬P)∧(R∨P∨N)∧(¬R∨¬L)∧(S)

Then comes the real DPLL:

```python title="pseudocode2 for DPLL"
function DPLL(clauses, symbols, model) returns true or false  
    # Early Termination ?
    if every clause in clauses is true in model then  
        return true  # 找到一个满足解   
    if some clause in clauses is false in model then  
        return false # 当前模型无法满足所有子句  

    # Pure Symbol Heuristic：查找在当前子句中只出现一次的符号  
    P, value ← FIND-PURE-SYMBOL(symbols, clauses, model)  
    if P is non-null then  
        # 如果找到，则将其赋值并继续递归  
        return DPLL(clauses, symbols - P, model ∪ {P = value})  

    # Unit Clause Heuristic：查找当前模型中只包含一个未赋值符号的子句  
    P, value ← FIND-UNIT-CLAUSE(clauses, model)  
    if P is non-null then  
        # 如果找到单子句，则将其赋值并继续递归  
        return DPLL(clauses, symbols - P, model ∪ {P = value})  

    # 分支：选择一个符号进行赋值  
    P ← FIRST(symbols); rest ← REST(symbols)  
    # 尝试将 P 赋值为 true，然后为 false  
    return DPLL(clauses, rest, model ∪ {P = true}) or  
           DPLL(clauses, rest, model ∪ {P = false})
```

First, we apply early termination: we check if given the current model, every clause is true, or at least one clause is false. Since the model hasn’t assigned any symbol yet, we don’t know which clauses are true or false yet. 

Next, we check for pure literals. There are no symbols that only appear in a non-negated form, or symbols that only appear in a negated form, so there are no pure literals that we can simplify. For example, N is not a pure literal because the first clause uses the negated ¬N, and the second clause uses the non-negated N. 

Next, we check for unit clauses (clauses with just one symbol). There’s one unit clause S. For this overall sentence to be true, we know that S has to be true (there’s no other way to satisfy that clause). Therefore, we can make another call to DPLL with S assigned to true in our model, and S removed from the list of symbols that still need assignments.

Our second DPLL call looks like this:

> [!INFO]
>
> - model: {S : T}
> - symbols: [L,M,N,P,Q,R]
> - clauses: (take S=T in) (¬N)∧(M∨Q∨N)∧(L∨¬M)∧(L∨¬Q)∧(¬L∨¬P)∧(R∨P∨N)∧(¬R∨ ¬L)

Then DPLL again, but when we check for unit clauses, there’s one unit clause (¬N). For this overall sentence to be true, (¬N) must be true, so N must be false.

> [!INFO]
>
> - model: {S : T,N : F}
> - symbols: [L,M,P,Q,R]
> - clauses:(take model in) (M∨Q)∧(L∨¬M)∧(L∨¬Q)∧(¬L∨¬P)∧(R∨P)∧(¬R∨¬L)

Go on, we pass three tricks and need to try to assign a value to a variable. From our fixed variable ordering, we’ll assign M first, and from our fixed value ordering, we’ll try making M true first[^7].

[^7]: Remember that we use a fixed variable ordering (alphabetical order) and a fixed value ordering (true before false).

> [!INFO]
>
> - model: {S : T,N : F,M : T}
> - symbols: [L,P,Q,R]
> - clauses:(take model in) (L)∧(L∨ ¬Q)∧(¬L∨ ¬P)∧(R∨P)∧(¬R∨ ¬L)

...

Finally, we get:

> [!INFO]
>
> - model: {S : T,N : F,M : F,Q : T,L : T,P : F}
> - symbols: [R]
> - clauses:(take model in) (R)∧(¬R)≡F

Obviously, the s is False if M is True. After the same operation, we found that the s is False even if M is False. Then we can conclude that this entire sentence is unsatisfiable, and we’re done.

### Theorem Proving

We could also prove entailment using three rules of inference:

1. If our knowledge base contains A and A ⇒ B we can infer B (Modus Ponens).
2. If our knowledge base contains A∧B we can infer A. We can also infer B. (And-Elimination).
3. If our knowledge base contains A and B we can infer A∧B (Resolution).

> [!KNOWLEDGE]- resolution algorithm
>
> The last rule forms the basis of the **resolution algorithm** which iteratively applies it to the knowledge base and to the newly inferred sentences until either q is inferred, in which case we have shown that KB ⊨ q, or there is nothing left to infer, in which case $KB \not\models q$. 
> 
> Although this algorithm is both **sound** (the answer will be correct) and **complete** (the answer will be found) it runs <u>in worst case time</u>  that is exponential in the size of the knowledge base[^8].

[^8]: 打个比方，我们想在 MC（应该没人不知道这是啥吧）制作一个蛋糕；作为一个萌新，我不知道合成途径，也不知道在工作台上我该如何摆放，甚至不知道我当前的版本能否合成蛋糕……我可以将所有方块任意组合/摆放来尝试合成，但是这样需要尝试的次数是很多的，远比 MC 中的方块种类多。

However, in the special case that our knowledge base only has literals (symbols by themselves) and implications: (P1∧···∧Pn ⇒ Q) ≡ (¬P1∨···∨¬P2∨Q), we can prove <u>entailment in time linear to the size of the knowledge base</u> . 

One algorithm, **forward chaining** iterates through every implication statement in which the **premise** (left hand side) is known to be true, adding the **conclusion** (right hand side) to the list of known facts. This is repeated until q is added to the list of known facts, or nothing more can be inferred.

```python title="pseudocode for PL-FC-Entails"
function PL-FC-ENTAILS?(KB, q) returns true or false  
	inputs:  KB, q

count ← table, where count[c] is the number of symbols in c’s premise  
inferred ← table, where inferred[s] is initially false for all symbols  
agenda ← a queue of symbols, initially symbols known to be true in KB  

while agenda is not empty do  
    p ← Pop(agenda)  # 从 agenda 中取出一个符号 p  
    if p = q then return true  # 如果 p 是查询 q，返回 true  
    if inferred[p] = false then  # 如果 p 尚未被推断  
        inferred[p] ← true  # 标记 p 为已推断  
        for each clause c in KB where p is in c.Premise do  
            decrement count[c]  # 对每个包含 p 的子句 c，减少其前提符号计数  
            if count[c] = 0 then  # 如果 c 的所有前提符号都已被推断  
                add c.Conclusion to agenda  # 将 c 的结论添加到 agenda  
return false  # 如果 agenda 处理完毕仍未找到 q，返回 false
```

### Forward Chaining: Example

Suppose we had the following knowledge base, We’d like to use forward chaining to determine if Q is true or false:

> [!EXAMPLE]
>
> 1. A → B
> 2. A → C
> 3. B∧C → D
> 4. D∧E → Q
> 5. A∧D → Q
> 6. A

To initialize the algorithm, we’ll initialize a list of numbers **count**. The ith number in the list tells us how many symbols are in the premise of the ith clause[^9]. 

[^9]: For example, the third clause B∧C → D has 2 symbols (B and C) in its premise, so the third number in our list should be 2. Note that the sixth clause A has 0 symbols in its premise, because it is equivalent to True → A.

Then, we’ll initialize **inferred**, a mapping of each symbol to true/false. This tells us which symbols we’ve found to be true. Initially, all symbols will be false, because we haven’t proven any symbols to be true yet. 

Finally, we’ll initialize a list of symbols **agenda**, which is a list of symbols that we can prove to be true, but have not propagated the effects of yet[^10]. Initially, agenda will only contain the symbols we directly know to be true, which is just A here. (In other words, agenda starts with any clauses with 0 symbols in its premise.)

[^10]:  For example, if D were in the agenda, this would indicate that we’re ready to prove that D is true, but we still need to check how that affects any of the other clauses.

> [!INFO]
>
> - count: [1,1,2,2,2,0]
> - inferred: {A : F,B : F,C : F,D : F,E : F,Q : F}
> - agenda: [A]

On each iteration, we’ll pop an element off agenda. Here, there’s only one element that we can pop off: A. The symbol we popped off is not the symbol we want to analyze (Q), so we’re not done with the algorithm yet. 

According to the inferred table, A is false. However, since we’ve just popped A off the agenda, we’re able to set it to true. 

Next, we need to propagate the consequences of A being true. For each clause where A is in the premise, we’ll decrement its corresponding count to indicate that there is one fewer symbol in the premise that needs to be checked. In this example, clauses 1, 2, and 5 contain A in the premise, so we’ll decrement elements 1, 2, and 5 in count. 

Finally, we check if any clauses have reached a count of 0, and add the conclusions of them to the agenda.

> [!INFO]
>
> - count: [0,0,2,2,1,0]
> - inferred: {A : T,B : F,C : F,D : F,E : F,Q : F}
> - agenda: [B,C]

Finally, we get:

> [!INFO]
>
> - count: [0,0,0,1,0,0]
> - inferred: {A : T,B : T,C : T,D : T,E : F,Q : F}
> - agenda: [Q]

Next, we’ll pop off Q from the agenda. This is the symbol we wanted to evaluate, and popping it off the agenda indicates that it has been proven to be true. We conclude that Q is true and finish the algorithm.

## link

- [cs188-sp24-note08](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/notes/cs188-sp24-note08.pdf)