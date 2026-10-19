import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("/tmp/fl2.xml")
d = mujoco.MjData(m)
q0 = m.jnt_qposadr[m.joint("boxj").id]
qp = m.jnt_qposadr[m.joint("pz").id]
mujoco.mj_forward(m, d)
for phase, frames in [("press", 200), ("climb", 320)]:
    for i in range(frames):
        # 目标 z 从 0.04 爬 16mm；x 预压在 XML 初值（1.5mm 过盈）
        d.ctrl[0] = (0.0001 * i if phase == "climb" else 0.0)
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    fz = fn = 0.0
    for ci in range(d.ncon):
        c = d.contact[ci]
        fr = np.array(c.frame).reshape(3,3)
        if abs(fr[0][0]) > 0.9:
            f = np.zeros((6,1)); mujoco.mj_contactForce(m, d, ci, f)
            fn += f[0][0]; fz += f[2][0]
    print(f"{phase}: box_z={(d.qpos[q0+2]-0.04)*1000:+.2f}mm pad_z={d.qpos[qp]*1000:.2f}mm padN={fn:.1f} Fz={fz:.2f}N")
print("lift:", f"{(d.qpos[q0+2]-0.04)*1000:+.2f} mm")
