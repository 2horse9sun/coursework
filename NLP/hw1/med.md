# 最小编辑距离

## 1 问题引入

**最小编辑距离**是从一个词变换到另一个词所需要的最少单字符操作总数（如插入，删除和替换）。其中，插入和删除的开销为1， 替换的开销为2。

例如，把单词`maximize`变为单词`minimum`的最小编辑距离为9，变换步骤如下：

```
Step 1 delete "a": "maximize" => "mximize"
Step 2 delete "x": "mximize" => "mimize"
Step 3 insert "n": "mimize" => "minmize"
Step 4 insert "i": "minmize" => "minimize"
Step 5 delete "i": "minimize" => "minimze"
Step 6 replace "z" with "u": "minimze" => "minimue"
Step 7 replace "e" with "m": "minimue" => "minimum"
```

## 2 计算模型

编辑距离问题拥有**最优子结构**，也就是说该问题能被分解成几个简单的子问题进行求解，子问题也能被分解为更简单的子问题，最后的子问题将很容易求解。下面给出编辑距离问题的描述及其子问题的描述，对不同的情形进行分析：

**问题描述**：通过对字符串`str1`进行编辑操作，把字符串`str1[1:m]`转换成字符串`str2[1:n]`。

**子问题描述**：通过对子串`str1`进行编辑操作，把子`str1[1:i]`转换成字符串`str2[1:j]`。

**情形1**：到达任一字串的末尾。

如果子串`str1`是空串，我们只需要把`str2`中剩余的字符都插入到`str1`当中即可，开销为`str2`中剩余字符的数量。如：

```
"", "ABC" => "ABC", "ABC"
cost = 3
```

如果`str2`是空串，结论同上。

**情形2**：子串`str1`和`str2`的最后一个字符相同。

如果出现以上情形，我们不需要进行任何编辑操作，开销为0。如：

```
"ACC", "ABC" => "AC", "AB"
cost = 0
```

**情形3**：子串`str1`和`str2`的最后一个字符不同。

在这种情况下，我们需要计算出以下三种操作的最小开销：

1. 把`str2`的最后一个字符插入到`str1`末尾。
2. 删除`str1`的最后一个字符。
3. 把`str1`的最后一个字符替换成`str2`的最后一个字符。

**状态转移方程**：
$$
T(i, j)=
\left\{  
             \begin{array}{**lr**}  
            \max(i,j), & \min(i,j)=0 \\  
            T(i-1,j-1), & str_1(i-1)=str_2(j-1)\\  
             \min\{T(i-1,j)+1,T(i,j-1)+1,T(i-1,j-1)+2\}, &    str_1(i-1)\neq str_2(j-1)
             \end{array}  
\right.
$$

## 3 编程实现



## 4 评估模型



