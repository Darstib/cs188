# shopSmart.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
Here's the intended output of this script, once you fill it in:

Welcome to shop1 fruit shop
Welcome to shop2 fruit shop
For orders:  [('apples', 1.0), ('oranges', 3.0)] best shop is shop1
For orders:  [('apples', 3.0)] best shop is shop2
"""
from __future__ import print_function
import shop


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


if __name__ == "__main__":
    "This code runs when you invoke the script from the command line"
    orders = [("apples", 1.0), ("oranges", 3.0)]
    dir1 = {"apples": 2.0, "oranges": 1.0}
    shop1 = shop.FruitShop("shop1", dir1)
    dir2 = {"apples": 1.0, "oranges": 5.0}
    shop2 = shop.FruitShop("shop2", dir2)
    shops = [shop1, shop2]
    print(
        "For orders ", orders, ", the best shop is", shopSmart(orders, shops).getName()
    )
    orders = [("apples", 3.0)]
    print(
        "For orders: ", orders, ", the best shop is", shopSmart(orders, shops).getName()
    )
