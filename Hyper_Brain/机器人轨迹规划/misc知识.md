- **URDF** 是一种用于描述机器人模型的XML格式语言

  ```xml
  <robot name="simple_arm">
    <!-- 底座连杆 -->
    <link name="base_link">
      <visual>
        <geometry><box size="0.2 0.2 0.1"/></geometry>
        <material name="gray"><color rgba="0.5 0.5 0.5 1"/></material>
      </visual>
    </link>
  
    <!-- 肩关节 -->
    <joint name="shoulder_joint" type="revolute">
      <parent link="base_link"/>
      <child link="upper_arm_link"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
    </joint>
  
    <!-- 大臂连杆 -->
    <link name="upper_arm_link">
      <visual>
        <geometry><cylinder length="0.3" radius="0.05"/></geometry>
        <material name="red"><color rgba="1 0 0 1"/></material>
      </visual>
    </link>
  </robot>
  ```

  1. **典型工具链**
     - **从 URDF 到网格模型**：通过 Blender 或 SolidWorks 导出为 STL/DAE 文件，再嵌入 URDF 的 `<visual>` 标签。
     - **运动学求解**：利用 ROS 的 `kdl_parser` 库解析 URDF 中的关节参数，计算机器人正逆运动学。
     - **物理仿真**：在 Gazebo 中加载 URDF 模型，结合 `<collision>` 和 `<inertial>` 参数实现动力学仿真。

- **SE3**表示一个物体在三维空间中的位姿（位置 + 方向）：

  ```python
  # 创建一个 SE(3) 对象
  T_world_link = jaxlie.SE3(
      wxyz=jnp.array([1.0, 0.0, 0.0, 0.0]),  # 单位四元数（无旋转）
      xyz=jnp.array([0.5, 0.3, 0.2])         # 平移 (x=0.5, y=0.3, z=0.2)
  )
  ```

  - 前向运动学（FK）：已知关节角 → 求末端执行器位置和方向。

  ```python
  Ts = robot.forward_kinematics(cfg)
  # 输入：cfg（关节角度）
  # 输出：每个连杆的 SE3 姿态（包括末端执行器）
  ```

  - 逆运动学（IK）或轨迹优化：已知目标 SE3 姿态 → 求满足该姿态的关节角度

  ```python
  sol_traj, sol_pos, sol_wxyz = pks.solve_online_planning(...)
  # 输入：末端目标 SE3
  # 输出：一组满足条件的关节角度
  ```

- 在EDMP中，数据的格式为：

  ```python
  @dataclass
  class PlanningProblem:
      target: SE3                # 目标末端位姿
      target_volume: Union[Cuboid, Cylinder]  # 目标区域体积
      q0: np.ndarray           # 起始关节角度
      obstacles: Optional[Obstacles] = None   # 障碍物列表
      obstacle_point_cloud: Optional[np.ndarray] = None
      target_negative_volumes: Obstacles = field(default_factory=lambda: [])
  ```

  ```python
  PlanningProblem(target=SE3(xyz=[0.07721892973355943, -0.4431428241168158, 0.4323637132137983], quaternion=[-0.12440869437591286, 0.7126188027223423, -0.6902976972437753, -0.013638473162225299]), target_volume=Cuboid(
      center=[0.07576218516603972, -0.4405667000925684, 0.42700732183432916],
      dims=[0.1, 0.1, 0.1],
      quaternion=[-0.12438362900094437, 0.7127279537071795, -0.6901929353105434, -0.013464356217371585],
  ), q0=array([ 0.99208685,  0.67555796, -2.23618836, -2.44863999, -0.00312386,
          2.30969288, -2.34159546]), obstacles=[Cuboid(
      center=[0.8078577973827096, -0.22577681593194587, 0.1642150186328124],
      dims=[1.044130653222579, 1.4644067583942613, 0.36843003726562484],
      quaternion=[1.0, 0.0, 0.0, 0.0],
  ), Cuboid(
      center=[0.15375694105255347, -0.6228134714555313, 0.1642150186328124],
      dims=[0.2640710594377333, 0.6703334473470904, 0.36843003726562484],
      quaternion=[1.0, 0.0, 0.0, 0.0],
  )], obstacle_point_cloud=None, target_negative_volumes=[])
  ```

  

