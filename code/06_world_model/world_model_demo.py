# -*- coding: utf-8 -*-
"""
世界模型最小落地 Demo：学习“状态转移”并在隐空间做多步推演

【环境依赖】
- Python 3.10+, PyTorch 2.x；CPU 可运行

【核心逻辑】
- 世界模型的核心是学 P(s_{t+1} | s_t, a_t)：给定当前状态与动作预测下一状态；
- 本 Demo 用“相位匀速旋转”的合成环境：状态 = [sin(θ), cos(θ)]，动作 = 角速度；
- 模型 = 小 MLP(state, action) -> next_state；
- 训练后做多步 rollout：只用第一步真值，后续全用模型预测，观察误差累积。

【关键参数】
- state_dim=2, action_dim=1, hidden=64, epochs=300, rollout_steps=20

【避坑】
1. 世界模型落地难在“状态从哪来”：真实场景需要编码器把观测压成状态；
2. 多步推演误差会累积，评测要看 rollout 长度而不是单步 loss；
3. 合成环境只是教学，真实世界模型需要大量交互数据。

【输出解读】
- 单步预测 loss 很小；
- 多步 rollout 前期贴合、后期发散属正常——误差累积是世界模型的核心难题。

【工程改造方向】
- 把状态换成图像 latent（加 VAE/编码器），就是视觉世界模型雏形；
- 用 rollout 误差做规划：选让未来误差最小的动作；
- 对接强化学习/具身场景做想象 rollout。
"""

import torch
import torch.nn as nn


def make_data(n=2000):
    """生成状态转移数据：θ_{t+1} = θ_t + ω。"""
    theta = torch.rand(n) * 6.28
    omega = torch.rand(n) * 0.5 + 0.2
    s = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
    a = omega.unsqueeze(-1)
    next_theta = theta + omega
    ns = torch.stack([torch.sin(next_theta), torch.cos(next_theta)], dim=-1)
    return s, a, ns


class DynamicsNet(nn.Module):
    """学 delta：next = s + f(s,a)，残差结构更容易学。"""

    def __init__(self, state_dim=2, action_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, s, a):
        return s + self.net(torch.cat([s, a], dim=-1))


def rollout(model, s0, a, steps=20):
    """只用第一步真值，后续全用模型自预测。"""
    s = s0
    traj = [s]
    with torch.no_grad():
        for _ in range(steps):
            s = model(s, a)
            traj.append(s)
    return torch.stack(traj)


def main():
    torch.manual_seed(0)
    s, a, ns = make_data()
    model = DynamicsNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    for step in range(300):
        pred = model(s, a)
        loss = loss_fn(pred, ns)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print("step %d loss %.5f" % (step, loss.item()))

    s0 = torch.tensor([[0.0, 1.0]])
    a0 = torch.tensor([[0.3]])
    traj = rollout(model, s0, a0, steps=20)
    print("rollout 20 步轨迹（应接近旋转圆上的点）:")
    print(traj[:5, 0].detach().numpy())


"""
进阶改造 Prompt
1. 增加动作对状态影响的非线性（如重力摆），观察模型是否学得动；
2. 加一个观测编码器（图像->latent），升级为视觉世界模型雏形；
3. 用 rollout 误差做简单规划：尝试多个动作序列选误差最小的；
4. 对比单步 loss 与 rollout 20 步误差，理解误差累积；
5. 调研自动驾驶/游戏世界模型工作（以官方发布为准）并复现其评测口径。
"""

if __name__ == "__main__":
    main()
