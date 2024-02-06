# 第三章标题

首先强化学习是一个序列决策问题，目标是获得长期收益的最大化，因此如何定义的我的优化目标了。

假设在一个时刻t的优化目标是我的吞吐量

$$
x=\alpha r
$$

其中r表示系统在时间t获得的吞吐量，$\alpha$表示因为干扰产生的性能下降因子。

r如何计算得来？

---

# 问题分析与建模

现在有一个边缘服务器的集合$N$,对于每个服务器$n$，可以提供的资源量有$\Lambda=(cpu,mem,net^{\uparrow},net^{\downarrow},disk^{read},disk^{write})$，除此以外，其存储空间为$C_n$

还有待部署的容器服务的集合$S$,他需要消耗的硬盘空间是$c_s$

假设一个容器服务$S$的请求$r_s$需要消耗的资源约为$\lambda_{s}=(cpu,mem,net^{\uparrow},net^{\downarrow},disk^{read},disk^{write})$

### 要解决的问题

首先是容器服务部署问题，容器既可以部署在边缘，也可以不部署在边缘。

容器服务的部署决策为

$$
x_{s,n}=\{0,1\}\qquad n\in N,\quad s\in S \qquad\qquad(3-1)
$$

他表示是否部署的到边缘服务器n上。

服务的请求路由问题，

$$
y_{r,n_0}=\{0,1\} \qquad\qquad r \in R,\quad n_0\in N\cup\{l_0\} \qquad (3-2)
$$

他表示服务请求$r$是由边缘服务器$n$来完成，还是由云数据中心$l$完成。

**默认情况下，用户只能请求到覆盖到他的边缘服务器**

### 问题本身的约束

请求必须由云数据中心或者边缘服务器完成

$$
\sum_{n_0\in N\cup\{l\}} y_{r,n_0}=1 ,\qquad\forall r\in R \qquad (3-3)
$$

### 服务完成约束

一个服务请求r由边缘节点n完成，那么边缘节点n必须要已经部署了服务s

$$
y_{r_s,n} \le x_{s,n} \quad \forall n \in N \qquad (3-4)
$$

### 资源约束

- 存储空间的约束，要想部署容器，边缘服务器上首先要有足够的存储空间,存储容器服务的镜像
- $$
  \sum x_{s,n} c_s \le C_n \qquad,\forall s\in S \qquad (3-5)
  $$
- 完成一个服务请求，边缘服务器需要提供足够的计算资源
- $$
  \sum_{r_s\in R} y_{r_s,n} \lambda_{r_s}^{cpu} \le \Lambda_{n}^{cpu} \quad\forall n\in N \qquad (3-6)
  $$
- 完成一个服务请求，边缘服务器提供足够的内存资源
- $$
  \sum_{r_s\in R} y_{r_s,n} \lambda_{r_s}^{mem} \le \Lambda_{n}^{mem} \quad\forall n\in N \qquad (3-7)
  $$
- 完成一个服务请求，边缘服务器提供足够的网络带宽资源
- $$
  \sum_{r_s\in R} y_{r_s,n} \lambda_{r_s}^{net^{\uparrow}} \le \Lambda_{n}^{net^{\uparrow}} \quad\forall n\in N \qquad (3-8)
  $$
- $$
  \sum_{r_s\in R} y_{r_s,n} \lambda_{r_s}^{net^{\downarrow}} \le \Lambda_{n}^{net^{\downarrow}} \quad\forall n\in N  \qquad (3-9)
  $$

  完成服务请求，边缘服务器需要提供足够的读写带宽资源
- $$
  \sum_{r_s\in R} y_{r_s,n} \lambda_{r_s}^{disk^{read}} \le \Lambda_{n}^{disk^{read}} \quad\forall n\in N \qquad(3-10)
  $$

  $$
  \sum_{r_s\in R} y_{r_s,n} \lambda_{r_s}^{disk^{write}} \le \Lambda_{n}^{disk^{write}} \quad\forall n\in N \qquad (3-11)
  $$

对于公式(3-6),(3-7),(3-8),(3-9),(3-10),(3-11)可以用一个线性方程组来表示即

$$
\begin{bmatrix}
\lambda^{cpu}_{r_s}\\
\lambda^{mem}_{r_s}\\
\lambda^{net^{\uparrow}}_{r_s}\\
\lambda^{net^{\downarrow}}_{r_s}\\
\lambda^{disk^{read}}_{r_s}\\
\lambda^{disk^{write}}_{r_s}\\
\end{bmatrix} y_{r_s,n} \le 
\begin{bmatrix}
\Lambda^{cpu}_{n}\\
\Lambda^{mem}_{n}\\
\Lambda^{net^{\uparrow}}_{n}\\
\Lambda^{net^{\downarrow}}_{n}\\
\Lambda^{disk^{read}}_{n}\\
\Lambda^{disk^{write}}_{n}\\
\end{bmatrix} \qquad\forall n\in N \qquad (3-12)
$$

公式(3-12)写成向量形式，可以写成

$$
\sum_{r_s \in S} y_{r_s,n} \vec{\lambda_{s}} \le \vec{\Lambda_n} \qquad \forall n \in N
$$

其中$\vec{\lambda_s}$是一个6维向量，表示服务请求$r_s$需要消耗的每种类型资源量。

在不用考虑干扰的情况下，请求路由和边缘服务部署问题的优化目标是最大化边缘服务器能完成的请求数量。

$$
Target=\min_{x,y} \quad\sum_{r\in R} y_{r,n} \qquad n\in N  \qquad (3-13)
\\s.t. \quad constraint(3-1)-(3-5),(3-12)
$$

在考虑干扰的情况下，优化目标应该是？

首先要计算一个容器$s$实际完成的服务请求数量数量,

要明确一点，一个服务可以部署在多个边缘服务器上。如何找到一个服务器n上容器s完成的请求数量了？

对于服务器n，遍历整个S集合，如果$x_{s,n}=1$,就表示服务器n上部署了容器s，再进一步计算容器s完成的请求数量。

进一步计算容器s完成的请求数量，要求请求类型为s，服务器为n，且$y_{r_s,n}=1$,即可计算在内。

因此，可以计算出服务器$n_0$上容器$s_0$完成的请求数量为：

$$
d_{n_0,s_0}=x_{n_0,s_0} \sum_{r_{s_0}\in S}  y_{r_{s_0},n_0} \quad n_0\in N
$$

相应的的服务器$n_0$上容器$s_0$的资源需求量$Q_{n_0,s_0}$,计算公式为，他其实是一个6维向量。

$$
Q_{n_0,s_0}=d_{n_0,s_0} \vec{\lambda_{s_0}}
$$

计算服务器$n_0$上容器$s_0$的干扰，还需要计算服务器$n_0$可以供给给容器$s_0$的空闲资源量$P_{n_0,s_0}$

$$
P_{n_0,s_0}=\vec{\Lambda_{n_0}}-\sum_{s \in S}y_{r_s,n_0}\vec{\lambda_s}+Q_{n_0,s_0}
$$

因此，对于部署在服务器$n_0$上的容器服务$s_0$的干扰程度为：

$$
\alpha_{n_0,s_0}=f_{danet}(Q_{n_0,s_0},P_{n_0,s_0})
$$

也就是对于部署在服务器$n_0$上的容器服务$s_0$其实际能够提供的吞吐量为，

$$
Throuput(n_0,s_0) = \alpha_{n_0,s_0}d_{n_0,s_0}
$$

整个边缘系统可以获得的全体吞吐量为：

$$
Target=\sum_{n_0 \in N} \sum_{s_0 \in S} Throuput(n_0,s_0) x_{n_0,s_0}
\\ = \sum_{n_0 \in N} \sum_{s_0 \in S} \alpha_{n_0,s_0}d_{n_0,s_0} x_{n_0,s_0}
\\= \sum_{n_0 \in N} \sum_{s_0 \in S} (\alpha_{n_0,s_0}x_{n_0,s_0} \sum_{r_{s_0}\in S} y_{r_{s_0},n_0})
\\= \sum_{n_0 \in N} \sum_{s_0 \in S} \alpha_{n_0,s_0}x_{n_0,s_0}y_{s_0,n_0}
$$

*其实不用化简，就用上面的公式就行了最终的优化目标为：*

$$
Target=\min_{x,y} Throuput(n_0,s_0) x_{n_0,s_0}
\\s.t. \quad constraint(3-1)-(3-5),(3-12)
$$

以上内容是t时刻的放置决定，本文的优化目标是获得长期收益。因此还要改进。

## 强化学习的要素建模

首先要把一个时间段划分成一组等间隔时间隙$\bold{T}={1,2,3,...,T}$，每个时间间隙的时常为$\Delta t$。

在每一个时隙$t\in \bold{T}$,我们假设每一个服务到达的请求速率为一个到达率为$b^{t}_l$的泊松过程。也就是说，在$\Delta t$之内，每个容器服务的请求数量固定为$b^{t}_l$。

每个请求$r_s$所需要的资源量服从指数分布?

### 状态空间

在每个时隙$t$的状态包括了三种信息

1. 上个时隙的服务放置情况
2. 当前时隙的请求到达情况
3. 上一个时隙的请求路由情况

其实第三点信息，对于我的这个部署来说可有可无，我只是想在每一个时隙获得一个部署方案，这个部署方案与上一个时隙的方案无关

### 动作空间

在每个时隙$t$,边缘节点需要做出服务部署和请求路由的联合决策，就是上面的所有的集合x,y

### 奖励函数 

奖励函数就是优化目标，

最小化就是负值，最大化就是正值。
