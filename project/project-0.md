---
tags:
  - notes
comments: true
dg-publish: true
---

# Tutorial

> [!PREREQUISITE]
>
> - [project 0](https://inst.eecs.berkeley.edu/~cs188/sp24/projects/proj0/)（若需要认证，可见[仓库](https://github.com/Darstib/cs188/tree/main/materials/project/intro_page)）

## explain

这个是让我们熟悉项目使用和 python 的考察，没啥好讲的。

```shell
python autograder.py # 评分全部
python autograder.py -q q1 # 测试第一问
```

```python title="addition.py"
def add(a, b):
    "Return the sum of a and b"
    "*** YOUR CODE HERE ***"
    return a + b
```

```python title="buyLotsOfFruit.py"
def buyLotsOfFruit(orderList):
    """
        orderList: List of (fruit, numPounds) tuples
    
    Returns cost of order
    """
    totalCost = 0.0
    "*** YOUR CODE HERE ***"
    # 对于这道简单的题，下面的条件语句倒是可以省略
    for fruit, numPounds in orderList:
        if fruit in fruitPrices:
            totalCost += fruitPrices[fruit] * numPounds
        else:
            print("Sorry we don't have %s" % fruit)
            return None
    return totalCost
```

```python title="shopSmart.py"
def shopSmart(orderList, fruitShops):
    """
    orderList: List of (fruit, numPound) tuples
    fruitShops: List of FruitShops
    """
    "*** YOUR CODE HERE ***"
    best_shop = None
    # 将 lowest_cost 初始化为正无穷大，这样第一次循环时，lowest_cost 会被更新为第一个商店的 cost
    lowest_cost = float("inf")
    for fruitShop in fruitShops:
        cost = 0
        for fruit, numPounds in orderList:
            cost += fruitShop.fruitPrices[fruit] * numPounds
        if cost < lowest_cost:
            lowest_cost = cost
            best_shop = fruitShop
    return best_shop
```
## pass

- [project-0 全部通过](attachments/project-0.png)
- [全代码](https://github.com/Darstib/cs188/tree/main/project/solution)