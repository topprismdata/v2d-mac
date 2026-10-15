import sys
import numpy as np
sys.path.insert(0, '.')
import mujoco
from retarget import retarget_smoke as rs

mj, model, data = rs.load_scene()
rs._mj_forward(model, data)
seq = rs.load_source_sequence()
joints21, _ = rs.mano_skeleton(seq)
retargeter, _ = rs.build_retargeter()
qpos = rs.retarget_sequence(joints21, retargeter)
anchor = rs.executed_anchor(qpos, model, data)
f_act = np.arange(6, 22)
w_act = np.arange(6)
lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
gf = np.clip(anchor["finger_cmd"][0], lo[f_act], hi[f_act])
gw = np.clip(anchor["ctrl"][0], lo[w_act], hi[w_act])
fq = rs._finger_qadr(model)
wq = np.array([model.jnt_qposadr[rs._joint_id(model, n)] for n in rs.WRIST_JOINTS])
frame_skip = max(1, int(round(1.0 / (rs.CTRL_HZ * model.opt.timestep))))
rs.reset_scene(model, data, fq, wq, f_act, w_act)
rs.warm_start(model, data, fq, wq, f_act, w_act, gf, gw,
              ctrl_hz=rs.CTRL_HZ, frame_skip=frame_skip)

names = rs.MJCF_FINGER_JOINTS
for step in range(2):
    for _ in range(frame_skip):
        mujoco.mj_step(model, data)
    print(f"--- after replay frame {step}")
    mj.mj_forward(model, data)
    for nm in ("mfj1", "mfj0", "ffj1", "mfj2"):
        c = names.index(nm)
        jid = rs._joint_id(model, nm)
        adr = int(model.jnt_dofadr[jid])
        qadr = int(model.jnt_qposadr[jid])
        a = rs._joint_actuator_id(model, nm)
        kp = float(model.actuator_gainprm[a, 0])
        M = float(model.dof_M0[adr] + model.dof_armature[adr])
        print(f"  {nm}: qpos={data.qpos[qadr]:.6f} ctrl={data.ctrl[a]:.6f} "
              f"err={data.qpos[qadr]-data.ctrl[a]:+.6e}")
        print(f"    actuator_force={data.actuator_force[a]:+.8f} kp*err={-kp*(data.qpos[qadr]-data.ctrl[a]):+.8f}")
        print(f"    qfrc_actuator={data.qfrc_actuator[adr]:+.8f}")
        print(f"    qfrc_constraint={data.qfrc_constraint[adr]:+.8f}")
        print(f"    qfrc_passive={data.qfrc_passive[adr]:+.8f}")
        print(f"    qfrc_bias={data.qfrc_bias[adr]:+.8f}")
        print(f"    M*qacc={M*data.qacc[adr]:+.8f} qacc={data.qacc[adr]:+.4f}")
        print(f"    qvel={data.qvel[adr]:+.6f}")
