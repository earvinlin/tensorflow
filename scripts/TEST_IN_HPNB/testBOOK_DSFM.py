"""
資料科學基礎數學
"""

from sympy import *

"""
# (p.8) sample 1-7 繪製線性函數
x = symbols('x')
f = 2*x + 1
plot(f)


# (p.9) sample 1-8 繪製指數函數
from sympy import *
x = symbols('x')
f = x**2 + 1
plot(f)


# (p.10) sample 1-9 宣告具有兩個自數的函數
from sympy.plotting import plot3d

x, y = symbols('x y')
f = 2*x + 3*y
plot3d(f)
"""

# (p.13) 
i, n = symbols('i n')

# 對從1 ~ n的所有i元素進行迭代，然後相乘並相加
summation = Sum(2*i,(i,1,n))

# 指明n為5，從數字1 ~ 5進行代
up_to_5 = summation.subs(n, 5)
print(up_to_5.doit())


# (p.15)
x = symbols('x')
expr = x**2 / x**5
print(expr)


