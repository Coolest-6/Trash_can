import mujoco
import os

# 把这里改成你的 urdf 文件名
urdf_path = "Handone.urdf"

# 加载 URDF（MuJoCo 会自动解析并转换）
model = mujoco.MjModel.from_xml_path(urdf_path)

# 保存为 MJCF XML 文件
mjcf_path = os.path.splitext(urdf_path)[0] + ".xml"
mujoco.mj_saveLastXML(mjcf_path, model)

print(f"转换完成，已保存为: {mjcf_path}")