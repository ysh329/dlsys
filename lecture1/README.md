# Lecture 1: A Crash Course on Deep Learning

## Elements of Machine Learning

First, we start from three machine learning keypoints as below:

- Model: a specific format, such as $y=kx+b$, $x$ and $y$ are features and label (class, annotation or others) respectively from one sample. However, both $w$ and $b$ are unknown for us (of course, we can initial them with some random values). To summarize, input $x_{i}$ in, class probability $\hat{y_{i}}$ out.
- Objective: an object to reflect the gap/difference of a model between perfect status and current status.
- Training: a procedure to minimize the gap. Due to stability of $x$ and $y$, the only changeable are $k$ and $b$.

![](./img/elements-of-ml.png)

On this page, we can see a dog picture as an input $x_i$, representing an feature vector. And after model transformation (the formulation to caculate y hat), we get the class probability of this picture.

