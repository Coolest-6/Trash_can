# Handone MuJoCo 学习与验证路线

这套教程以当前大拇指模型为最小实验平台。核心原则是：先证明模型可信，再验证控制器，最后训练强化学习策略。否则算法可能只是在利用错误的质量、碰撞或约束。

## 统一实验流程

每项实验都按四步进行：**明确假设 → 运行可重复脚本 → 记录量化指标 → 与 CAD、理论或实物对照**。所有角度在 MuJoCo 内部使用 rad，长度使用 m，力矩使用 N·m。

## 阶段 0：模型可信度基线（当前阶段）

目标是理解 `MjModel`、`MjData`、`qpos`、`qvel`、`ctrl` 和 `mj_step`，并确认模型能稳定重复运行。

- `MjModel`：从 XML 编译出的只读机械系统参数。
- `MjData`：某一时刻的状态、控制量、力和传感器结果。
- `qpos`：广义位置；本模型四个转动关节对应四个角度。
- `qvel`：广义速度，单位 rad/s。
- `ctrl`：给执行器的命令；当前 `<position>` 执行器中是目标角度。
- `mj_forward`：已知状态时计算位姿和力，不推进时间。
- `mj_step`：根据动力学把仿真向前推进一个 `timestep`。

先运行：

```powershell
D:\miniconda\envs\mujoco_env\python.exe tutorials/01_model_audit.py
D:\miniconda\envs\mujoco_env\python.exe tutorials/02_state_and_motion.py
```

验收：模型成功加载；零位闭环误差小于 0.1 mm；运动中无 NaN、明显穿透或持续振荡；清楚区分机械关节限位与执行器软件限位。

## 阶段 1：机械设计校核

逐项校核关节轴线、零位、运动范围、闭环装配误差、自碰撞、质量、质心和惯量。随后加入规范的碰撞几何，做工作空间、机械干涉、峰值速度/加速度、驱动力矩和约束反力扫描。结果与 SolidWorks Motion 或手算量级对比。

验收不是“看起来能动”，而是坐标误差、质量误差、最大干涉量、峰值力矩等均有表格和容差。需要特别注意：STL 三角网格适合显示，不一定适合稳定接触；碰撞通常应使用简化 primitive 或 convex mesh。

## 阶段 2：控制算法验证（重点讲解）

按难度依次实现：

1. 位置 P/PD 控制：理解误差、刚度、阻尼、超调、稳态误差和饱和。
2. 轨迹生成：用梯形速度或五次多项式代替角度突变。
3. 前馈控制：加入重力/惯性补偿，减少反馈负担。
4. 状态机或阻抗控制：自由运动用位置控制，接触后控制“柔软程度”和接触力。

重点指标：上升时间、超调量、稳态误差、峰值力矩、闭环点误差和抗扰恢复时间。建议提前理解导数、二阶弹簧阻尼系统、PID、采样周期、饱和与 anti-windup；状态空间只需先理解“状态决定系统未来”。

## 阶段 3：小规模强化学习（重点讲解）

先构造一个很小的任务，例如“控制两个电机让指尖到目标点”，不要一开始做完整抓取。环境定义包含：

- observation：关节角、关节速度、目标位置、接触信息；
- action：目标角增量或力矩；初学阶段推荐目标角增量；
- reward：距离奖励、成功奖励，以及动作、速度、碰撞惩罚；
- termination：成功、超时、越界或仿真异常；
- reset/domain randomization：随机目标，并逐步随机质量、摩擦和控制延迟。

先学会 MDP 的状态、动作、奖励、转移、折扣回报，再理解 policy、value 和 PPO 的直觉即可。神经网络反向传播不必先手推。算法代码、并行训练、调参、曲线和模型保存由项目脚本负责。

验收：与随机策略、固定控制器比较；分离训练/测试目标；报告成功率而非只看累计奖励；固定随机种子保证可复现。

## 阶段 4：轨迹数据采集

统一记录时间、目标角、实际角/速度、执行器力、指尖位姿、接触力、约束误差和事件标记。先保存 CSV 便于检查，批量训练再使用 NPZ/HDF5。每份数据同时保存模型版本、步长、随机种子和实验参数。

## 阶段 5：接触力仿真

加入刚性测试物体和指腹碰撞体，标定摩擦、接触刚度/阻尼与求解步长。区分接触法向力、切向摩擦力、执行器力矩和闭环约束力。用简单实验校准：已知载荷静压、斜面滑动、不同速度压入；之后再做抓取。接触力结果只有在材料、碰撞几何和求解器参数经过校准后才有工程意义。

## 建议学习顺序

当前只需掌握 Python 基础、NumPy 数组、坐标系/齐次变换、rad 与 degree、速度和加速度。进入控制阶段前学习二阶系统与 PD；进入强化学习前学习 MDP 和 PPO 的概念。每进入下一阶段，先完成上一阶段验收，不以“窗口里能运行”作为通过标准。

## 当前实验：阶跃与五次轨迹

运行：

```powershell
D:\miniconda\envs\mujoco_env\python.exe tutorials/03_compare_trajectories.py
```

两组实验具有相同起点、目标角和仿真时间。`step` 在第一个时间步直接给出最终目标；`quintic` 使用：

```text
s(r) = 10r^3 - 15r^4 + 6r^5,  r = t / T
q_target(t) = q_start + s(r)(q_end - q_start)
```

其中 `r` 被限制在 0 到 1。该函数在起终点的速度和加速度均为零，因此不会人为制造无限大的理论速度和加速度。观察终端中的峰值速度、峰值加速度、峰值执行器力和峰值闭环误差；详细时序数据保存在 `tutorials/results/03_trajectory_comparison.csv`。

注意：`max tracking error` 不能直接用于判断两种方法谁更好。平滑轨迹有意让目标逐渐变化，因此运动初期与最终目标存在距离。更有意义的是峰值速度、加速度、力矩、闭环误差，以及轨迹结束后的最终误差。

## 当前实验：闭环机构全范围扫描

运行：

```powershell
D:\miniconda\envs\mujoco_env\python.exe tutorials/04_linkage_range_scan.py
```

脚本将 `finger1_motor` 保持在零位，让 `finger3_motor` 以五次轨迹从最小角缓慢扫描到最大角，再反向返回。每个采样点记录四个关节角、速度、执行器力矩、闭环误差、碰撞及 `bigfinger4_Link` 原点的世界坐标。结果保存在 `tutorials/results/04_linkage_range_scan.csv`。

重点理解：

- `q2/q4 range` 是两个被动关节随主动关节产生的运动范围。
- `dq2/du`、`dq4/du` 表示主动角变化 1 rad 时被动角大约变化多少 rad；数值突然变大可能提示接近特殊构型。
- `hysteresis` 是相同指令角在正向和反向扫描中的最大关节角差。理想刚性、无摩擦、无碰撞的准静态机构应非常小。
- 负的 `contact distance` 表示碰撞几何发生穿透。
- 执行器力达到 `forcerange` 上限表示饱和，后续角度不一定能够继续准确跟踪。

当前模型约在 `35.75°` 出现 `base_link` 与 `bigfinger2_Link` 接触，并在高角度区出现力矩饱和和明显滞回。在确认该接触是实际机械干涉还是 STL/碰撞建模造成的假接触之前，不能把完整 `0–44.69°` 范围都认定为可靠工作区。

复核 CAD 装配关系后，已确认该处没有真实干涉。MuJoCo 对网格碰撞使用凸化后的形状，可能把 STL 的凹槽或避让空间填实，因此正式模型仅排除了 `base_link` 与 `bigfinger2_Link` 这一组已确认的假接触。排除后全范围扫描无接触，最大闭环误差约 `0.031 mm`，最大往返差约 `0.012°`。

可视化原始凸包接触：

```powershell
D:\miniconda\envs\mujoco_env\python.exe tutorials/05_contact_debug_viewer.py
```

红色标记和箭头表示 MuJoCo 报告的接触点与接触力。对比正式模型：

```powershell
D:\miniconda\envs\mujoco_env\python.exe tutorials/05_contact_debug_viewer.py --production
```

正式模型中的 `thumb_tip` 位于 `bigfinger3_Link` 局部坐标 `(0.03718, 0.01251, 0.00332) m`。后续工作空间与接触实验都以该 site 作为指尖测量点。

## 当前实验：二维驱动下的三维指尖工作空间

运行：

```powershell
D:\miniconda\envs\mujoco_env\python.exe tutorials/06_thumb_workspace_scan.py
```

脚本将两个电机的软件角度范围各离散为21点，扫描 `21 × 21 = 441` 个组合。每个目标通过五次轨迹到达并等待稳定，然后记录指尖世界坐标、四个关节角、执行器力、闭环误差、接触和数值雅可比。

输出：

- `tutorials/results/06_thumb_workspace.csv`：每个构型的完整数值。
- `tutorials/results/06_thumb_workspace_projections.svg`：工作空间的 XY、XZ、YZ 投影。

雅可比 `J` 描述“小关节角变化会造成多大的指尖位置变化”：

```text
delta_tip ≈ J * delta_joint
```

脚本对 `J` 做奇异值分解。最小奇异值接近零，表示指尖在某个方向几乎无法运动；条件数越大，表示不同方向的运动能力越不均匀，控制和逆运动学越敏感。

当前扫描441个构型全部通过可靠性检查。指尖范围约为 X `42.59..100.28 mm`、Y `-80.90..-18.01 mm`、Z `61.16..100.88 mm`；最大跟踪误差约 `0.071°`，最大闭环误差约 `0.0089 mm`，无碰撞。最差雅可比条件数约 `1.86`，当前范围内没有明显运动学奇异点。

为了避免位置伺服与机械硬限位互相对抗，`bigfinger3_joint` 的机械上限保持 `0.78 rad`，`finger3_motor` 软件上限使用 `0.77 rad`，保留约 `0.57°` 安全裕量。求解步长使用 `0.5 ms`；较旧的 `1 ms` 配置在高角度构型下会使 `finger1_motor` 出现离散时间振荡。

## 当前实验：指尖逆运动学

运行内置的三个可达目标：

```powershell
D:\miniconda\envs\mujoco_env\python.exe tutorials/07_inverse_kinematics.py
```

求解自定义世界坐标目标，单位为毫米：

```powershell
D:\miniconda\envs\mujoco_env\python.exe tutorials/07_inverse_kinematics.py --target-mm 75 -55 85
```

显示求解后的往复运动：

```powershell
D:\miniconda\envs\mujoco_env\python.exe tutorials/07_inverse_kinematics.py --viewer
```

求解器先从工作空间表中选择距离目标最近的角度作为初值，再反复执行：计算当前位置误差、计算数值雅可比、使用阻尼最小二乘求角度修正、限制单次步长并检查角度边界。阻尼项可以避免接近奇异构型时角度修正异常增大。

`predicted error` 是工作空间插值模型中的误差；`MuJoCo error` 是把求得角度送回完整闭环动力学模型后的实际误差，后者是最终验收依据。误差不超过 `0.25 mm`、无碰撞且数值有效时标记为 `reachable=True`。不可达目标会返回工作空间内的最近结果，并保留非零残差，而不会假装求解成功。

当前三个内置目标的MuJoCo误差约为 `0.026 mm`、`0.023 mm` 和 `0.009 mm`，均通过验证。结果保存到 `tutorials/results/07_inverse_kinematics_results.csv`。

## 当前实验：质量属性审计

运行：

```powershell
D:\miniconda\envs\mujoco_env\python.exe tutorials/08_mass_properties_audit.py
```

脚本从URDF读取质量、质心和含惯性积的完整惯量矩阵，再把MuJoCo的主惯量与主轴四元数重建为link局部坐标系下的矩阵进行比较。同时检查惯量矩阵是否正定以及是否满足刚体惯量三角关系。

五个刚体总质量在URDF和MJCF中均为 `0.17585972 kg`；所有质心完全一致，惯量矩阵相对误差约为 `10^-7..10^-6`，全部通过物理合法性检查。`base_link` 原先由MuJoCo按网格自动估算质量，现已显式补回URDF中的 `0.11298 kg` 和完整惯量。

审计还会报告关节定义差异，但不会自动修改。当前定制MJCF与源URDF存在三项差异：`bigfinger1_joint` 上限 `0.85 vs 0.78 rad`、`bigfinger3_joint` 轴向 `+Z vs -Z`、`bigfinger4_joint` 范围 `[-1.5,1.5] vs [-0.5,0.5] rad`。当前闭环和控制结果均以MJCF定义为准，这些差异需要保留设计来源说明。

## 当前实验：运动时间与动态载荷

运行：

```powershell
D:\miniconda\envs\mujoco_env\python.exe tutorials/09_dynamic_load_scan.py
```

两个电机执行相同的 `0 -> 0.70 rad` 五次轨迹，只改变动作完成时间。每次测试先静置到闭环平衡，再记录四个关节的峰值速度/加速度、两个执行器的峰值力矩/功率/做功、饱和比例、闭环误差和最终误差。

按“力矩低于上限90%、速度不超过URDF上限、闭环误差不超过0.1 mm、最终误差不超过0.1°”验收，当前测试中 `0.20 s` 是最快合格动作；`0.10 s` 和 `0.15 s` 分别因闭环误差约 `0.232 mm` 和 `0.121 mm` 未通过。详细结果位于 `tutorials/results/09_dynamic_load_scan.csv`。

该结果只是当前仿真模型下的比较，不等于真实电机选型结论。加入电机额定/峰值力矩、减速比、转子惯量、速度限制、驱动器电流限制和热约束后，需要重新执行扫描。
