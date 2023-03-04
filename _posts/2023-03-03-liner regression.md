---
title: "tensorflow로 liner regression 학습 1"
date: "2023-03-04"
categories: [Deep learning]
tags: [tensorflow]
---


Cost and Hypothesis

<img src="./images/2023-03-03/2023-03-04-i1.png" width=140>

2.5와 0.5를 각각 초기값으로 지정

```
x_data = [1,2,3,4,5]<br>
y_data = [1,2,3,4,5]<br>
<br>
w = tf.Variable(2.9)<br>
b = tf.Variable(0.5)<br>
<br>
hypothesis = x_data * w + b
```

cost/loss function

```
cost = tf.reduce_mean(tf.square(hypothesis - y_data))
```

learning rate 0.01로 설정

```
learning_rate=0.01
```


for 을 활용하여 Gradient Descent 계산하기

```python
for i in range(100+1):<br>     
    with tf.GradientTape() as tape:<br>  
        hypothesis = w * x_data +b<br>  
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))<br>  
        w_grad, b_grad = tape.gradient(cost, [w,b])<br>  
        w.assign_sub(learning_rate * w_grad)<br>  
        b.assign_sub(learning_rate * b_grad)<br>    
        if i % 10 ==0:<br>  
            print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, w.numpy(), b.numpy(), cost))
```






[def]: ./images/hc.png